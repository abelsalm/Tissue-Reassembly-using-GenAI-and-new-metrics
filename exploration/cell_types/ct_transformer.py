"""Gene-only cell-type set transformer.

Mirror of ``models/{model,transformer,self_attention}.py`` with every
position and diffusion-time path removed. Cells attend to each other using
transcriptome embeddings only; output is per-cell class logits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import LinearAttentionTransformer
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from utils.data.dataholder import DataHolder

# Names accepted in config / constructor for the gene-MLP nonlinearities.
GENE_MLP_ACTIVATIONS = ("relu", "none", "gelu", "silu")


class Identity(nn.Module):
    """No-op activation (``gene_mlp_activation: \"none\"``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_activation(name: str) -> nn.Module:
    """Map a config string to an activation module.

    Accepted (case-insensitive): ``relu``, ``none`` / ``identity`` /
    ``no_activation``, ``gelu``, ``silu`` / ``swish``.
    """
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "relu": "relu",
        "none": "none",
        "identity": "none",
        "no_activation": "none",
        "noactivation": "none",
        "gelu": "gelu",
        "silu": "silu",
        "swish": "silu",
    }
    if key not in aliases:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Choose one of: {', '.join(GENE_MLP_ACTIVATIONS)}"
        )
    canon = aliases[key]
    if canon == "relu":
        return nn.ReLU()
    if canon == "none":
        return Identity()
    if canon == "gelu":
        return nn.GELU()
    if canon == "silu":
        return nn.SiLU()
    raise ValueError(f"Unhandled activation '{name}'")


def _resolve_gene_mlp_activations(
    gene_mlp_activation: Union[str, Sequence[str]],
) -> tuple[str, str]:
    """Normalize to a pair ``(act_after_first_linear, act_after_second_linear)``."""
    if isinstance(gene_mlp_activation, (list, tuple)):
        if len(gene_mlp_activation) != 2:
            raise ValueError(
                "gene_mlp_activation list must have length 2 "
                f"(got {len(gene_mlp_activation)}): {gene_mlp_activation}"
            )
        return str(gene_mlp_activation[0]), str(gene_mlp_activation[1])
    name = str(gene_mlp_activation)
    return name, name


class GeneSelfAttention(nn.Module):
    """Same idea as ``models.self_attention.SelfAttention``, genes only.

    Original does: Linear(X) ‖ pos_embed ‖ time_embed → Linear → linear-attn.
    Here: Linear(X) → linear-attn (no concat, no pos/time heads).
    """

    def __init__(self, node_features_dimensions: int, num_heads: int) -> None:
        super().__init__()
        assert node_features_dimensions % num_heads == 0, (
            f"dx={node_features_dimensions} must be divisible by "
            f"num_heads={num_heads}"
        )
        self.node_features_dimensions = node_features_dimensions
        self.num_heads = num_heads

        self.lin_node_features = nn.Linear(
            node_features_dimensions, node_features_dimensions
        )
        self.attention = LinearAttentionTransformer(
            dim=node_features_dimensions,
            heads=num_heads,
            depth=1,
            max_seq_len=70000,
        )

    def forward(
        self, node_features: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        # node_features: (B, N, dx); node_mask: (B, N)
        x = self.lin_node_features(node_features)
        x = self.attention(x)
        x = x * node_mask.unsqueeze(-1).to(x.dtype)
        return x


class GeneTransformerLayer(nn.Module):
    """Same residual + FFN block as ``models.transformer.TransformerLayer``,
    but only the node-feature branch (no PositionNorm / time FFN).
    """

    def __init__(
        self,
        node_features_dimensions: int,
        num_heads: int,
        dim_ff_node_features: int = 384,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = GeneSelfAttention(
            node_features_dimensions=node_features_dimensions,
            num_heads=num_heads,
        )

        self.lin_node_features_1 = Linear(
            node_features_dimensions, dim_ff_node_features
        )
        self.lin_node_features_2 = Linear(
            dim_ff_node_features, node_features_dimensions
        )
        self.norm_node_features_1 = LayerNorm(
            node_features_dimensions, eps=layer_norm_eps
        )
        self.norm_node_features_2 = LayerNorm(
            node_features_dimensions, eps=layer_norm_eps
        )
        self.dropout_node_features_1 = Dropout(dropout)
        self.dropout_node_features_2 = Dropout(dropout)
        self.dropout_node_features_3 = Dropout(dropout)
        self.activation = F.relu

    def forward(
        self, node_features: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_out = self.self_attn(node_features, node_mask)

        x = self.dropout_node_features_1(attn_out)
        x = self.norm_node_features_1(node_features + x)

        ff = self.lin_node_features_2(
            self.dropout_node_features_2(self.activation(self.lin_node_features_1(x)))
        )
        ff = self.dropout_node_features_3(ff)
        x = self.norm_node_features_2(x + ff)
        x = x * node_mask.unsqueeze(-1).to(x.dtype)
        return x


def _normalize_global_skip(global_skip: Optional[str]) -> Optional[str]:
    """Return ``'concat'``, ``'add'``, or ``None`` (disabled)."""
    if global_skip is None or global_skip is False:
        return None
    if isinstance(global_skip, str):
        key = global_skip.strip().lower()
        if key in ("", "none", "off", "false", "0"):
            return None
        if key in ("concat", "cat", "concatenate"):
            return "concat"
        if key in ("add", "sum", "plus"):
            return "add"
    raise ValueError(
        f"global_skip must be null/none, 'concat', or 'add' (got {global_skip!r})"
    )


def masked_mean_pool(
    x: torch.Tensor, node_mask: torch.Tensor
) -> torch.Tensor:
    """Mean over real cells only → ``(B, 1, D)``."""
    mask = node_mask.unsqueeze(-1).to(dtype=x.dtype)  # (B, N, 1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (x * mask).sum(dim=1, keepdim=True) / denom


class CellTypeTransformer(nn.Module):
    """Gene MLP → gene-only transformer stack → optional fusions → logits.

    Optional features (config / constructor):
      * ``global_skip``: ``None`` | ``\"concat\"`` | ``\"add\"`` — bridge the
        unmixed gene-MLP embedding ``E_raw`` past the transformer so the
        classifier always sees intrinsic identity (anti over-smoothing).
      * ``subgraph_summary``: if True, masked mean-pool of ``E_mixed`` is
        broadcast and concatenated as a bag-level neighborhood descriptor.

    Only ``node_features`` and ``node_mask`` are used; positions / time ignored.
    """

    def __init__(
        self,
        gene_dim: int,
        num_classes: int,
        n_layers: int = 4,
        hidden_mlp_dims: Optional[dict] = None,
        hidden_dims: Optional[dict] = None,
        dropout_cls: float = 0.1,
        dropout_layer: float = 0.1,
        gene_mlp_activation: Union[str, Sequence[str]] = "relu",
        global_skip: Optional[str] = None,
        subgraph_summary: bool = False,
    ) -> None:
        super().__init__()
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"X": 384}
        if hidden_dims is None:
            hidden_dims = {"dx": 64, "num_heads": 16, "dim_ffX": 384}

        self.gene_dim = gene_dim
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.dx = int(hidden_dims["dx"])
        self.global_skip = _normalize_global_skip(global_skip)
        self.subgraph_summary = bool(subgraph_summary)

        act1_name, act2_name = _resolve_gene_mlp_activations(gene_mlp_activation)
        self.gene_mlp_activation = (act1_name, act2_name)
        act1 = build_activation(act1_name)
        act2 = build_activation(act2_name)

        # Same layout as ``models.model.Model.mlp_in_node_features``, with
        # configurable nonlinearities after each Linear.
        self.mlp_in_node_features = nn.Sequential(
            nn.Linear(gene_dim, hidden_mlp_dims["X"]),
            act1,
            nn.Linear(hidden_mlp_dims["X"], self.dx),
            act2,
        )

        self.transformer_layers = nn.ModuleList(
            [
                GeneTransformerLayer(
                    node_features_dimensions=self.dx,
                    num_heads=int(hidden_dims["num_heads"]),
                    dim_ff_node_features=int(hidden_dims["dim_ffX"]),
                    dropout=dropout_layer,
                )
                for _ in range(n_layers)
            ]
        )

        head_in = self._head_input_dim()
        self.head_input_dim = head_in
        # Last Linear → ``num_classes`` logits (not probabilities, not one-hot).
        # Train with integer-index CE in ``ct_losses.cross_entropy``.
        self.cls_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, hidden_mlp_dims["X"]),
            nn.ReLU(),
            nn.Dropout(dropout_cls),
            nn.Linear(hidden_mlp_dims["X"], num_classes),
        )

    def _head_input_dim(self) -> int:
        """Channel count into ``cls_head`` given skip / summary options."""
        if self.global_skip == "concat":
            dim = 2 * self.dx
        else:
            # ``add`` or disabled: cell stream stays ``dx``.
            dim = self.dx
        if self.subgraph_summary:
            dim += self.dx
        return dim

    def encode_raw_and_mixed(
        self, data: DataHolder
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(E_raw, E_mixed)``, each ``(B, N, dx)``, masked."""
        node_mask = data.node_mask
        e_raw = self.mlp_in_node_features(data.node_features)
        e_raw = e_raw * node_mask.unsqueeze(-1).to(e_raw.dtype)

        e_mixed = e_raw
        for layer in self.transformer_layers:
            e_mixed = layer(e_mixed, node_mask)
        return e_raw, e_mixed

    def fuse_for_head(
        self,
        e_raw: torch.Tensor,
        e_mixed: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optional global skip + subgraph summary → head features."""
        if self.global_skip == "concat":
            cell_feat = torch.cat([e_raw, e_mixed], dim=-1)
        elif self.global_skip == "add":
            cell_feat = e_raw + e_mixed
        else:
            cell_feat = e_mixed

        if self.subgraph_summary:
            # Bag-level mean of transformer states (pads ignored), broadcast.
            summary = masked_mean_pool(e_mixed, node_mask)  # (B, 1, dx)
            summary = summary.expand(-1, cell_feat.size(1), -1)
            cell_feat = torch.cat([cell_feat, summary], dim=-1)

        cell_feat = cell_feat * node_mask.unsqueeze(-1).to(cell_feat.dtype)
        return cell_feat

    def encode(self, data: DataHolder) -> torch.Tensor:
        """Return features fed to the classifier ``(B, N, head_input_dim)``."""
        e_raw, e_mixed = self.encode_raw_and_mixed(data)
        return self.fuse_for_head(e_raw, e_mixed, data.node_mask)

    def forward(self, data: DataHolder) -> torch.Tensor:
        """
        Args:
            data: ``node_features`` ``(B, N, G)``, ``node_mask`` ``(B, N)``.
                ``positions`` / ``diffusion_time`` are ignored if present.

        Returns:
            logits ``(B, N, num_classes)`` (padded rows zeroed).
        """
        x = self.encode(data)
        logits = self.cls_head(x)
        logits = logits * data.node_mask.unsqueeze(-1).to(logits.dtype)
        return logits
