"""Feature-only slice context and context-conditioned LUNA decoding.

The context encoder deliberately accepts no position or diffusion-time input.
Its output can therefore be cached and reused across every reverse-diffusion
step for a selected domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch
import torch.nn as nn

from exploration.cell_types.ct_transformer import GeneTransformerLayer
from models.model import Model
from utils.data.dataholder import DataHolder


@dataclass(frozen=True)
class SliceContext:
    """Cached output of :class:`SliceContextEncoder`.

    ``memory`` starts with the requested-domain and global-slice query tokens,
    followed by one state per full-slice cell.
    """

    memory: torch.Tensor
    memory_mask: torch.Tensor
    cell_states: torch.Tensor
    requested_domain_state: torch.Tensor
    global_slice_state: torch.Tensor
    target_membership: torch.Tensor
    target_domain_id: torch.Tensor


class SliceContextEncoder(nn.Module):
    """Coordinate-free, masked full-slice gene transformer."""

    def __init__(
        self,
        hidden_dim: int,
        num_domains: int,
        *,
        n_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        domain_summary_dim: int = 0,
        attention_type: str = "linear",
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_domains < 1:
            raise ValueError("num_domains must be positive")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )

        self.hidden_dim = int(hidden_dim)
        self.num_domains = int(num_domains)
        self.attention_type = _validate_attention_type(attention_type)
        self.role_embedding = nn.Embedding(2, self.hidden_dim)
        # Domain IDs in this repository reserve zero for UNK / padding.
        self.requested_domain_embedding = nn.Embedding(
            self.num_domains + 1, self.hidden_dim, padding_idx=0
        )
        self.requested_domain_query = nn.Parameter(
            torch.empty(1, 1, self.hidden_dim)
        )
        self.global_slice_query = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        nn.init.normal_(self.requested_domain_query, std=0.02)
        nn.init.normal_(self.global_slice_query, std=0.02)

        self.cell_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.domain_summary_projection: Optional[nn.Linear]
        if domain_summary_dim > 0:
            self.domain_summary_projection = nn.Linear(
                int(domain_summary_dim), self.hidden_dim
            )
        else:
            self.domain_summary_projection = None

        if self.attention_type == "linear":
            self.layers = nn.ModuleList(
                [
                    GeneTransformerLayer(
                        node_features_dimensions=self.hidden_dim,
                        num_heads=num_heads,
                        dim_ff_node_features=dim_feedforward,
                        dropout=dropout,
                        attention_use_node_mask=True,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    MaskedMultiheadSelfAttention(
                        hidden_dim=self.hidden_dim,
                        num_heads=num_heads,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        node_mask: torch.Tensor,
        target_membership: torch.Tensor,
        target_domain_id: torch.Tensor,
        requested_domain_summary: Optional[torch.Tensor] = None,
    ) -> SliceContext:
        """Encode one requested domain per full slice.

        Parameters use only biological features and masks. In particular, this
        method has no coordinate or timestep argument by construction.
        """
        if cell_embeddings.dim() != 3:
            raise ValueError(
                "cell_embeddings must be (B, Nslice, D), got "
                f"{tuple(cell_embeddings.shape)}"
            )
        batch_size, n_cells, dim = cell_embeddings.shape
        if dim != self.hidden_dim:
            raise ValueError(f"embedding dim {dim} != hidden_dim={self.hidden_dim}")
        mask = _normalize_cell_mask(node_mask, batch_size, n_cells, "node_mask")
        membership = _normalize_cell_mask(
            target_membership, batch_size, n_cells, "target_membership"
        )
        membership = membership & mask
        if not bool(membership.any(dim=1).all()):
            raise ValueError("each batch element must contain target-domain cells")
        domain_id = _normalize_target_domain_id(target_domain_id, batch_size)
        if bool(((domain_id < 1) | (domain_id > self.num_domains)).any()):
            raise ValueError(
                f"target_domain_id must be in [1, {self.num_domains}]"
            )

        role = self.role_embedding(membership.long())
        cells = self.cell_fusion(torch.cat([cell_embeddings, role], dim=-1))
        cells = cells * mask.unsqueeze(-1).to(cells.dtype)

        requested = (
            self.requested_domain_query.expand(batch_size, -1, -1)
            + self.requested_domain_embedding(domain_id).unsqueeze(1)
        )
        if requested_domain_summary is not None:
            if self.domain_summary_projection is None:
                raise ValueError(
                    "requested_domain_summary was supplied but domain_summary_dim=0"
                )
            if requested_domain_summary.dim() == 3:
                if requested_domain_summary.size(1) != 1:
                    raise ValueError(
                        "requested_domain_summary must be (B, D) or (B, 1, D)"
                    )
                requested_domain_summary = requested_domain_summary.squeeze(1)
            if requested_domain_summary.size(0) != batch_size:
                raise ValueError("requested_domain_summary batch size mismatch")
            requested = requested + self.domain_summary_projection(
                requested_domain_summary
            ).unsqueeze(1)

        global_query = self.global_slice_query.expand(batch_size, -1, -1)
        x = torch.cat([requested, global_query, cells], dim=1)
        query_mask = torch.ones(
            batch_size, 2, dtype=torch.bool, device=cell_embeddings.device
        )
        memory_mask = torch.cat([query_mask, mask], dim=1)
        for layer in self.layers:
            x = layer(x, memory_mask)

        return SliceContext(
            memory=x,
            memory_mask=memory_mask,
            cell_states=x[:, 2:],
            requested_domain_state=x[:, 0],
            global_slice_state=x[:, 1],
            target_membership=membership,
            target_domain_id=domain_id,
        )


class MaskedLinearCrossAttention(nn.Module):
    """Linear-complexity cross-attention with masked context keys/values.

    This follows the softmax factorization used by the repository's
    ``linear_attention_transformer`` dependency: queries normalize over head
    features, keys normalize over sequence positions, then ``K^T V`` is formed
    before applying queries.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        ff_dim = int(dim_feedforward or 4 * hidden_dim)

        self.norm_query = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm_context = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.to_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_states: torch.Tensor,
        context_memory: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_states.dim() != 3 or context_memory.dim() != 3:
            raise ValueError("target_states and context_memory must be rank-3")
        if target_states.size(0) != context_memory.size(0):
            raise ValueError("target and context batch sizes differ")
        if target_states.size(-1) != self.hidden_dim:
            raise ValueError("target_states hidden dimension mismatch")
        if context_memory.size(-1) != self.hidden_dim:
            raise ValueError("context_memory hidden dimension mismatch")

        batch_size, n_target, _ = target_states.shape
        n_context = context_memory.size(1)
        memory_mask = _normalize_cell_mask(
            context_mask, batch_size, n_context, "context_mask"
        )
        if target_mask is None:
            query_mask = torch.ones(
                batch_size, n_target, dtype=torch.bool, device=target_states.device
            )
        else:
            query_mask = _normalize_cell_mask(
                target_mask, batch_size, n_target, "target_mask"
            )

        q = self._split_heads(self.to_query(self.norm_query(target_states)))
        normalized_context = self.norm_context(context_memory)
        k = self._split_heads(self.to_key(normalized_context))
        v = self._split_heads(self.to_value(normalized_context))

        # Same non-causal linear attention factorization as LUNA.
        q = torch.softmax(q, dim=-1) * (self.head_dim ** -0.5)
        key_fill = torch.finfo(k.dtype).min
        key_mask = memory_mask[:, None, :, None]
        k = torch.softmax(k.masked_fill(~key_mask, key_fill), dim=-2)
        v = v.masked_fill(~key_mask, 0.0)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        attended = torch.einsum("bhnd,bhde->bhne", q, kv)
        attended = attended.transpose(1, 2).reshape(
            batch_size, n_target, self.hidden_dim
        )
        attended = self.attention_dropout(self.to_output(attended))
        x = target_states + attended
        x = x + self.feedforward(self.norm_ff(x))
        return x * query_mask.unsqueeze(-1).to(x.dtype)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        return x.view(
            batch_size, length, self.num_heads, self.head_dim
        ).transpose(1, 2)


class MaskedMultiheadSelfAttention(nn.Module):
    """Quadratic full attention option for slices small enough to fit in memory."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, states: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask.bool()
        normalized = self.norm_attention(states)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )
        states = states + attended
        states = states + self.feedforward(self.norm_feedforward(states))
        return states * mask.unsqueeze(-1).to(states.dtype)


class MaskedMultiheadCrossAttention(nn.Module):
    """PyTorch MultiheadAttention alternative to linear cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ff_dim = int(dim_feedforward or 4 * hidden_dim)
        self.norm_query = nn.LayerNorm(hidden_dim)
        self.norm_context = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_feedforward = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        target_states: torch.Tensor,
        context_memory: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory_mask = context_mask.bool()
        query_mask = (
            torch.ones(
                target_states.shape[:2],
                dtype=torch.bool,
                device=target_states.device,
            )
            if target_mask is None
            else target_mask.bool()
        )
        attended, _ = self.attention(
            self.norm_query(target_states),
            self.norm_context(context_memory),
            self.norm_context(context_memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        states = target_states + attended
        states = states + self.feedforward(self.norm_feedforward(states))
        return states * query_mask.unsqueeze(-1).to(states.dtype)


class DomainContextModel(nn.Module):
    """Wrapper pairing the legacy LUNA target decoder with cached slice context."""

    def __init__(
        self,
        input_dims,
        n_layers: int,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims,
        *,
        context_cfg=None,
        positionMLP_eps: float = 1e-9,
        embedding_cfg=None,
        num_cell_types: int = 0,
        num_domains: int = 0,
        train_node_features: Optional[torch.Tensor] = None,
        train_cell_type: Optional[torch.Tensor] = None,
        build_cell_type_mds: bool = False,
    ) -> None:
        super().__init__()
        if num_domains < 1:
            raise ValueError("DomainContextModel requires num_domains >= 1")
        self.target_decoder = Model(
            input_dims=input_dims,
            n_layers=n_layers,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            positionMLP_eps=positionMLP_eps,
            embedding_cfg=embedding_cfg,
            num_cell_types=num_cell_types,
            num_domains=num_domains,
            train_node_features=train_node_features,
            train_cell_type=train_cell_type,
            build_cell_type_mds=build_cell_type_mds,
        )

        hidden_dim = int(hidden_dims["dx"])
        num_heads = int(_cfg_value(context_cfg, "num_heads", hidden_dims["num_heads"]))
        domain_summary_dim = (
            int(self.target_decoder.domain_embedding.n)
            if self.target_decoder.domain_embedding is not None
            else 0
        )
        self.context_encoder = SliceContextEncoder(
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            n_layers=int(_cfg_value(context_cfg, "n_layers", 2)),
            num_heads=num_heads,
            dim_feedforward=int(
                _cfg_value(context_cfg, "dim_feedforward", hidden_dims["dim_ffX"])
            ),
            dropout=float(_cfg_value(context_cfg, "dropout", 0.1)),
            domain_summary_dim=domain_summary_dim,
            attention_type=str(
                _cfg_value(context_cfg, "encoder_attention_type", "linear")
            ),
        )
        self.target_input_fusion = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        cross_layers = _cross_attention_layer_indices(context_cfg, n_layers)
        cross_ff = int(
            _cfg_value(context_cfg, "cross_attention_dim_feedforward", hidden_dims["dim_ffX"])
        )
        cross_dropout = float(
            _cfg_value(context_cfg, "cross_attention_dropout", 0.1)
        )
        cross_attention_type = _validate_attention_type(
            _cfg_value(context_cfg, "cross_attention_type", "linear")
        )
        cross_attention_cls = (
            MaskedLinearCrossAttention
            if cross_attention_type == "linear"
            else MaskedMultiheadCrossAttention
        )
        self.cross_attention_blocks = nn.ModuleDict(
            {
                str(index): cross_attention_cls(
                    hidden_dim,
                    num_heads,
                    dim_feedforward=cross_ff,
                    dropout=cross_dropout,
                )
                for index in cross_layers
            }
        )

    def encode_context(
        self,
        context_data: DataHolder,
        target_domain_id: torch.Tensor,
        target_membership: Optional[torch.Tensor] = None,
    ) -> SliceContext:
        """Encode biological full-slice context once, with no coordinates/time."""
        if context_data.node_features is None or context_data.node_mask is None:
            raise ValueError("context_data requires node_features and node_mask")
        if context_data.domain_id is None:
            raise ValueError("context_data.domain_id is required")
        features = context_data.node_features
        batch_size, n_cells = features.shape[:2]
        mask = _normalize_cell_mask(
            context_data.node_mask, batch_size, n_cells, "context node_mask"
        )
        domain_ids = _normalize_domain_ids(
            context_data.domain_id, batch_size, n_cells
        )
        requested = _normalize_target_domain_id(target_domain_id, batch_size)
        expected_membership = mask & (domain_ids == requested.unsqueeze(1))
        if target_membership is None:
            membership = expected_membership
        else:
            membership = _normalize_cell_mask(
                target_membership, batch_size, n_cells, "target_membership"
            )
            if not torch.equal(membership & mask, expected_membership):
                raise ValueError(
                    "target_membership must select all and only cells in the "
                    "single requested target domain"
                )
        if not bool(expected_membership.any(dim=1).all()):
            raise ValueError(
                "each batch element must request exactly one non-empty target domain"
            )

        cell_embeddings, domain_table = self.target_decoder.encode_node_features(
            features,
            mask,
            cell_type=context_data.cell_type,
            domain_id=domain_ids,
            return_domain_table=True,
        )
        requested_summary = None
        if domain_table is not None:
            gather_index = requested.view(batch_size, 1, 1).expand(
                -1, 1, domain_table.size(-1)
            )
            requested_summary = domain_table.gather(1, gather_index).squeeze(1)
        return self.context_encoder(
            cell_embeddings,
            mask,
            membership,
            requested,
            requested_domain_summary=requested_summary,
        )

    def decode_target(
        self,
        target_data: DataHolder,
        context: SliceContext,
        target_to_context_indices: torch.Tensor,
    ) -> DataHolder:
        """Denoise target rows using a reusable encoded full-slice context."""
        if target_data.node_features is None or target_data.node_mask is None:
            raise ValueError("target_data requires node_features and node_mask")
        batch_size, n_target = target_data.node_features.shape[:2]
        if batch_size != context.cell_states.size(0):
            raise ValueError("target and context batch sizes differ")
        indices = target_to_context_indices
        if indices.dim() == 3 and indices.size(-1) == 1:
            indices = indices.squeeze(-1)
        if tuple(indices.shape) != (batch_size, n_target):
            raise ValueError(
                "target_to_context_indices must be (B, Ntarget), got "
                f"{tuple(indices.shape)}"
            )
        target_mask = _normalize_cell_mask(
            target_data.node_mask, batch_size, n_target, "target node_mask"
        )
        n_context = context.cell_states.size(1)
        valid_indices = (indices >= 0) & (indices < n_context)
        if not bool((valid_indices | ~target_mask).all()):
            raise ValueError("real target cells have out-of-range context indices")
        safe_indices = indices.long().clamp(0, max(n_context - 1, 0))
        gathered_membership = context.target_membership.gather(1, safe_indices)
        if not bool((gathered_membership | ~target_mask).all()):
            raise ValueError(
                "target_to_context_indices includes non-target context cells"
            )
        gather = safe_indices.unsqueeze(-1).expand(
            -1, -1, context.cell_states.size(-1)
        )
        target_context = context.cell_states.gather(1, gather)
        biological = self.target_decoder.encode_node_features(
            target_data.node_features,
            target_mask,
            cell_type=target_data.cell_type,
            domain_id=target_data.domain_id,
        )
        fused = self.target_input_fusion(
            torch.cat([biological, target_context], dim=-1)
        )
        fused = fused * target_mask.unsqueeze(-1).to(fused.dtype)
        return self.target_decoder(
            target_data,
            precomputed_node_embedding=fused,
            context_memory=context.memory,
            context_mask=context.memory_mask,
            cross_attention_blocks=self.cross_attention_blocks,
        )

    def forward(self, data: DataHolder) -> DataHolder:
        """Legacy-compatible unconditioned path."""
        return self.target_decoder(data)

    def cell_type_mds_loss(self) -> torch.Tensor:
        return self.target_decoder.cell_type_mds_loss()

    @property
    def cell_type_embedding(self):
        return self.target_decoder.cell_type_embedding

    @property
    def domain_embedding(self):
        return self.target_decoder.domain_embedding

    @staticmethod
    def modify_context(
        context: SliceContext,
        *,
        drop_mask: Optional[torch.Tensor] = None,
        shuffle_cells: bool = False,
    ) -> SliceContext:
        """Create dropped/shuffled ablation memory without re-encoding a slice."""
        memory = context.memory
        cell_states = context.cell_states
        if shuffle_cells:
            shuffled = []
            for batch_index in range(cell_states.size(0)):
                permutation = torch.randperm(
                    cell_states.size(1), device=cell_states.device
                )
                shuffled.append(cell_states[batch_index, permutation])
            cell_states = torch.stack(shuffled, dim=0)
            memory = torch.cat([memory[:, :2], cell_states], dim=1)
        if drop_mask is not None:
            drop = drop_mask.bool().reshape(-1, 1, 1)
            memory = memory.masked_fill(drop, 0.0)
            cell_states = cell_states.masked_fill(drop, 0.0)
            requested = context.requested_domain_state.masked_fill(
                drop.squeeze(1), 0.0
            )
            global_state = context.global_slice_state.masked_fill(
                drop.squeeze(1), 0.0
            )
        else:
            requested = memory[:, 0]
            global_state = memory[:, 1]
        return SliceContext(
            memory=memory,
            memory_mask=context.memory_mask,
            cell_states=cell_states,
            requested_domain_state=requested,
            global_slice_state=global_state,
            target_membership=context.target_membership,
            target_domain_id=context.target_domain_id,
        )

    def load_legacy_decoder_state_dict(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict:
        """Load shape-compatible legacy ``Model`` weights into the decoder.

        Lightning's optional ``model.`` prefix is accepted. Newly introduced
        context parameters remain freshly initialized and are reported missing.
        """
        source = state_dict.get("state_dict", state_dict)
        decoder_state = self.target_decoder.state_dict()
        compatible = {}
        skipped = {}
        for raw_name, value in source.items():
            name = str(raw_name)
            if name.startswith("model."):
                name = name[len("model.") :]
            if name.startswith("target_decoder."):
                name = name[len("target_decoder.") :]
            if name in decoder_state and decoder_state[name].shape == value.shape:
                compatible[name] = value
            else:
                skipped[str(raw_name)] = tuple(value.shape)
        result = self.target_decoder.load_state_dict(compatible, strict=False)
        return {
            "loaded": sorted(compatible),
            "missing": sorted(result.missing_keys),
            "skipped": skipped,
        }


def _normalize_cell_mask(
    mask: torch.Tensor, batch_size: int, n_cells: int, name: str
) -> torch.Tensor:
    if mask.dim() == 3 and mask.size(-1) == 1:
        mask = mask.squeeze(-1)
    if tuple(mask.shape) != (batch_size, n_cells):
        raise ValueError(
            f"{name} must be ({batch_size}, {n_cells}), got {tuple(mask.shape)}"
        )
    return mask.bool()


def _normalize_domain_ids(
    domain_ids: torch.Tensor, batch_size: int, n_cells: int
) -> torch.Tensor:
    if domain_ids.dim() == 3 and domain_ids.size(-1) == 1:
        domain_ids = domain_ids.squeeze(-1)
    if tuple(domain_ids.shape) != (batch_size, n_cells):
        raise ValueError(
            "domain_id must be "
            f"({batch_size}, {n_cells}), got {tuple(domain_ids.shape)}"
        )
    return domain_ids.long()


def _normalize_target_domain_id(
    target_domain_id: torch.Tensor, batch_size: int
) -> torch.Tensor:
    domain_id = target_domain_id.long()
    while domain_id.dim() > 1 and domain_id.size(-1) == 1:
        domain_id = domain_id.squeeze(-1)
    if domain_id.dim() == 0 and batch_size == 1:
        domain_id = domain_id.unsqueeze(0)
    if tuple(domain_id.shape) != (batch_size,):
        raise ValueError(
            "exactly one target_domain_id is required per batch element; "
            f"expected ({batch_size},), got {tuple(domain_id.shape)}"
        )
    return domain_id


def _cross_attention_layer_indices(context_cfg, n_layers: int) -> list[int]:
    configured = _cfg_value(context_cfg, "cross_attention_layers", None)
    if configured is None:
        every = int(_cfg_value(context_cfg, "cross_attention_every", 1))
        if every < 1:
            raise ValueError("cross_attention_every must be >= 1")
        return list(range(every - 1, n_layers, every))
    indices = sorted({int(index) for index in configured})
    if any(index < 0 or index >= n_layers for index in indices):
        raise ValueError(
            f"cross_attention_layers must use zero-based indices in [0, {n_layers})"
        )
    return indices


def _cfg_value(cfg, name: str, default):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        value = cfg.get(name, default)
    else:
        value = getattr(cfg, name, default)
    return default if value is None else value


def _validate_attention_type(value) -> str:
    key = str(value).strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear",
        "linear_attention": "linear",
        "multihead": "multihead",
        "multiheadattention": "multihead",
        "multi_head_attention": "multihead",
        "mha": "multihead",
    }
    if key not in aliases:
        raise ValueError(
            "attention type must be 'linear' or 'multihead' "
            f"(got {value!r})"
        )
    return aliases[key]
