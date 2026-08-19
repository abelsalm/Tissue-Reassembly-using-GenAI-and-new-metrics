"""Gene-only cell-type set transformer.

Mirror of ``models/{model,transformer,self_attention}.py`` with every
position and diffusion-time path removed. Cells attend to each other using
transcriptome embeddings only; outputs are per-cell class logits and
(optionally) soft cell-microenvironment logits over the same ``C`` classes.
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
    depth: int = 2,
) -> tuple[str, ...]:
    """Normalize to one activation name per gene-MLP linear (length ``depth``)."""
    if isinstance(gene_mlp_activation, (list, tuple)):
        if len(gene_mlp_activation) == depth:
            return tuple(str(a) for a in gene_mlp_activation)
        if len(gene_mlp_activation) == 2 and depth == 2:
            return str(gene_mlp_activation[0]), str(gene_mlp_activation[1])
        raise ValueError(
            f"gene_mlp_activation list must have length {depth} "
            f"(got {len(gene_mlp_activation)}): {gene_mlp_activation}"
        )
    name = str(gene_mlp_activation)
    return tuple(name for _ in range(depth))


def _build_gene_mlp(
    gene_dim: int,
    gene_hidden_dim: int,
    dx: int,
    depth: int,
    activation_names: Sequence[str],
) -> nn.Sequential:
    """Stack ``depth`` linears: G → [gene_hidden]* → dx with activations after each."""
    if depth < 2:
        raise ValueError(f"gene_mlp_depth must be >= 2 (got {depth})")
    if len(activation_names) != depth:
        raise ValueError(
            f"Expected {depth} activation names, got {len(activation_names)}"
        )
    in_dims = [gene_dim] + [gene_hidden_dim] * (depth - 1)
    out_dims = [gene_hidden_dim] * (depth - 1) + [dx]
    layers: list[nn.Module] = []
    for in_d, out_d, act_name in zip(in_dims, out_dims, activation_names):
        layers.append(nn.Linear(in_d, out_d))
        layers.append(build_activation(act_name))
    return nn.Sequential(*layers)


class GeneSelfAttention(nn.Module):
    """Same idea as ``models.self_attention.SelfAttention``, genes only.

    Original does: Linear(X) ‖ pos_embed ‖ time_embed → Linear → linear-attn.
    Here: Linear(X) → linear-attn (no concat, no pos/time heads).
    """

    def __init__(
        self,
        node_features_dimensions: int,
        num_heads: int,
        use_node_mask: bool = False,
    ) -> None:
        super().__init__()
        assert node_features_dimensions % num_heads == 0, (
            f"dx={node_features_dimensions} must be divisible by "
            f"num_heads={num_heads}"
        )
        self.node_features_dimensions = node_features_dimensions
        self.num_heads = num_heads
        self.use_node_mask = bool(use_node_mask)

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
        if self.use_node_mask:
            x = x * node_mask.unsqueeze(-1).to(x.dtype)
            x = self.attention(x, input_mask=node_mask.bool())
        else:
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
        ffn_activation: str = "relu",
        attention_use_node_mask: bool = False,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = bool(pre_norm)
        self.self_attn = GeneSelfAttention(
            node_features_dimensions=node_features_dimensions,
            num_heads=num_heads,
            use_node_mask=attention_use_node_mask,
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
        self.activation = build_activation(ffn_activation)

    def forward(
        self, node_features: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pre_norm:
            x_n = self.norm_node_features_1(node_features)
            attn_out = self.self_attn(x_n, node_mask)
            x = node_features + self.dropout_node_features_1(attn_out)
            x_n2 = self.norm_node_features_2(x)
            ff = self.lin_node_features_2(
                self.dropout_node_features_2(
                    self.activation(self.lin_node_features_1(x_n2))
                )
            )
            x = x + self.dropout_node_features_3(ff)
        else:
            attn_out = self.self_attn(node_features, node_mask)
            x = self.dropout_node_features_1(attn_out)
            x = self.norm_node_features_1(node_features + x)
            ff = self.lin_node_features_2(
                self.dropout_node_features_2(
                    self.activation(self.lin_node_features_1(x))
                )
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


def masked_max_pool(
    x: torch.Tensor, node_mask: torch.Tensor
) -> torch.Tensor:
    """Max over real cells only → ``(B, 1, D)`` (pads filled with large negative)."""
    mask = node_mask.unsqueeze(-1).to(dtype=x.dtype)  # (B, N, 1)
    fill = torch.finfo(x.dtype).min
    return x.masked_fill(mask == 0, fill).max(dim=1, keepdim=True).values


def gene_knn_mean_pool(
    e_raw: torch.Tensor,
    e_mixed: torch.Tensor,
    node_mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Per-cell mean of gene-KNN neighbors' mixed embeddings ``(B, N, D)``.

    Similarity from L2-normalized ``e_raw`` (gene MLP output); values from
    ``e_mixed``. Pads are ignored. Self is excluded from the neighbor set.
    """
    if k <= 0:
        raise ValueError(f"gene_knn_k must be > 0 (got {k})")
    bsz, n_cells, dim = e_mixed.shape
    mask = node_mask.bool()
    # Cosine sim in gene-embedding space.
    q = F.normalize(e_raw, dim=-1)
    sim = torch.bmm(q, q.transpose(1, 2))  # (B, N, N)
    sim = sim.masked_fill(~mask.unsqueeze(1), float("-inf"))
    sim = sim.masked_fill(~mask.unsqueeze(2), float("-inf"))
    # Exclude self.
    eye = torch.eye(n_cells, device=sim.device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(eye, float("-inf"))
    # Cap k by number of real neighbors available (approx via N-1).
    kk = min(k, max(1, n_cells - 1))
    idx = sim.topk(kk, dim=-1).indices  # (B, N, k)
    batch_ix = torch.arange(bsz, device=e_mixed.device).view(bsz, 1, 1).expand(
        bsz, n_cells, kk
    )
    neigh = e_mixed[batch_ix, idx]  # (B, N, k, D)
    # Invalidate pads / non-finite similarity slots (empty neighborhoods).
    valid = mask.unsqueeze(1).expand(-1, n_cells, -1).gather(2, idx)
    sim_vals = sim.gather(2, idx)
    valid = valid & torch.isfinite(sim_vals)
    weights = valid.to(dtype=e_mixed.dtype).unsqueeze(-1)
    denom = weights.sum(dim=2).clamp_min(1.0)
    out = (neigh * weights).sum(dim=2) / denom
    return out * mask.unsqueeze(-1).to(dtype=out.dtype)


class MaskedAttentionPool(nn.Module):
    """Learned attention pool over real cells → ``(B, 1, D)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, N, D); node_mask: (B, N)
        scores = self.score(x).squeeze(-1)  # (B, N)
        mask = node_mask.to(dtype=x.dtype)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, N, 1)
        weights = weights * mask.unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denom
        return (x * weights).sum(dim=1, keepdim=True)


def _mlp_head(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    activation: nn.Module,
    dropout: float,
) -> nn.Sequential:
    """LayerNorm → Linear → act → Dropout → Linear."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        activation,
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _stage1_tower(
    in_dim: int,
    out_dim: int,
    activation: nn.Module,
    dropout: float,
) -> nn.Sequential:
    """Trunk → intermediate task representation (no class logits)."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, out_dim),
        activation,
        nn.Dropout(dropout),
    )


class CellTypeTransformer(nn.Module):
    """Gene MLP → gene-only transformer stack → optional fusions → logits.

    Optional features (config / constructor):
      * ``global_skip``: ``None`` | ``\"concat\"`` | ``\"add\"`` — bridge the
        unmixed gene-MLP embedding ``E_raw`` past the transformer so the
        classifier always sees intrinsic identity (anti over-smoothing).
      * ``subgraph_summary``: if True, pool ``E_mixed`` (``subgraph_pool``:
        ``\"mean\"`` | ``\"mean_max\"`` | ``\"attn\"``) and broadcast-concatenate
        as a bag-level neighborhood descriptor.
      * ``feature_cross``: if True (requires ``predict_cme``), stage-1 type/env
        towers produce intermediates ``t_i``, ``e_i``; fuse with optional ``h_i``
        via LN+MLP; stage-2 heads read the fused features for final logits.
        Stage-1 linear probes are also returned for ablation.
      * ``cme_presence_gate``: if True (requires ``predict_cme``), the CME head
        splits into presence (sigmoid) and composition (softmax); the final
        mix is ``normalize(π ⊙ q)`` returned as ``cme_probs`` (and log-probs
        as ``cme_logits`` for SoftCE-compatible callers that already softmax —
        prefer ``cme_probs`` in the loss).
      * ``gene_mlp_depth``: number of gene-MLP linears (2 = G→gene_hidden→dx;
        3 adds an extra gene_hidden block).
      * ``layer_activation``: FFN nonlinearity inside each ``GeneTransformerLayer``.

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
        dropout_cme: float = 0.1,
        dropout_fusion: float = 0.1,
        gene_mlp_activation: Union[str, Sequence[str]] = "relu",
        gene_mlp_depth: int = 2,
        layer_activation: str = "relu",
        cls_activation: str = "relu",
        cme_activation: str = "relu",
        fusion_activation: Optional[str] = None,
        global_skip: Optional[str] = None,
        subgraph_summary: bool = False,
        subgraph_pool: str = "mean",
        subgraph_pool_source: str = "mixed",
        attention_use_node_mask: bool = False,
        cme_detach_type_in_fusion: bool = False,
        dual_fusion: bool = False,
        gene_knn_k: int = 0,
        pre_norm: bool = False,
        cme_condition_on_cls: bool = False,
        predict_cme: bool = False,
        feature_cross: bool = False,
        feature_cross_include_h: bool = True,
        cme_presence_gate: bool = False,
        presence_bias_init: float = 0.0,
        presence_topk: int = 0,
        cme_entropy_temp_t0: float = 1.0,
        cme_entropy_temp_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_mlp_dims is None:
            hidden_mlp_dims = {"gene": 384, "cls": 384}
        if hidden_dims is None:
            hidden_dims = {"dx": 64, "num_heads": 16, "dim_ffX": 384}

        # Backward-compatible keys: ``X`` used to size both MLPs.
        if "gene" not in hidden_mlp_dims and "X" in hidden_mlp_dims:
            hidden_mlp_dims = {
                **hidden_mlp_dims,
                "gene": hidden_mlp_dims["X"],
            }
        if "cls" not in hidden_mlp_dims:
            hidden_mlp_dims = {
                **hidden_mlp_dims,
                "cls": hidden_mlp_dims.get(
                    "gene", hidden_mlp_dims.get("X", 384)
                ),
            }
        if "cme" not in hidden_mlp_dims:
            hidden_mlp_dims = {
                **hidden_mlp_dims,
                "cme": hidden_mlp_dims.get(
                    "cls", hidden_mlp_dims.get("gene", 384)
                ),
            }

        self.gene_dim = gene_dim
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.dx = int(hidden_dims["dx"])
        self.gene_hidden_dim = int(hidden_mlp_dims["gene"])
        self.cls_hidden_dim = int(hidden_mlp_dims["cls"])
        self.cme_hidden_dim = int(hidden_mlp_dims["cme"])
        # Presence gate is a lighter twin of the CME composition head by default.
        self.presence_hidden_dim = int(
            hidden_mlp_dims.get("presence", min(128, self.cme_hidden_dim))
        )
        self.global_skip = _normalize_global_skip(global_skip)
        self.subgraph_summary = bool(subgraph_summary)
        pool_key = str(subgraph_pool).strip().lower()
        if pool_key not in ("mean", "mean_max", "attn"):
            raise ValueError(
                f"subgraph_pool must be 'mean', 'mean_max', or 'attn' "
                f"(got {subgraph_pool!r})"
            )
        self.subgraph_pool = pool_key
        src_key = str(subgraph_pool_source).strip().lower()
        if src_key not in ("mixed", "raw"):
            raise ValueError(
                f"subgraph_pool_source must be 'mixed' or 'raw' "
                f"(got {subgraph_pool_source!r})"
            )
        self.subgraph_pool_source = src_key
        self.attention_use_node_mask = bool(attention_use_node_mask)
        self.subgraph_attn_pool: Optional[MaskedAttentionPool] = None
        if self.subgraph_summary and self.subgraph_pool == "attn":
            self.subgraph_attn_pool = MaskedAttentionPool(self.dx)
        self.predict_cme = bool(predict_cme)
        self.feature_cross = bool(feature_cross)
        self.feature_cross_include_h = bool(feature_cross_include_h)
        self.cme_presence_gate = bool(cme_presence_gate)
        self.presence_bias_init = float(presence_bias_init)
        self.presence_topk = int(presence_topk)
        self.cme_detach_type_in_fusion = bool(cme_detach_type_in_fusion)
        self.dual_fusion = bool(dual_fusion)
        self.gene_knn_k = int(gene_knn_k)
        if self.gene_knn_k < 0:
            raise ValueError(f"gene_knn_k must be >= 0 (got {gene_knn_k})")
        self.pre_norm = bool(pre_norm)
        self.cme_condition_on_cls = bool(cme_condition_on_cls)
        self.cme_entropy_temp_t0 = float(cme_entropy_temp_t0)
        self.cme_entropy_temp_alpha = float(cme_entropy_temp_alpha)

        if self.feature_cross and not self.predict_cme:
            raise ValueError("feature_cross=True requires predict_cme=True")
        if self.cme_presence_gate and not self.predict_cme:
            raise ValueError("cme_presence_gate=True requires predict_cme=True")

        self.gene_mlp_depth = int(gene_mlp_depth)
        gene_act_names = _resolve_gene_mlp_activations(
            gene_mlp_activation, depth=self.gene_mlp_depth
        )
        self.gene_mlp_activation = gene_act_names
        self.layer_activation = str(layer_activation)
        self.cls_activation = str(cls_activation)
        self.cme_activation = str(cme_activation)
        fus_act_name = (
            str(fusion_activation)
            if fusion_activation is not None
            else str(cls_activation)
        )
        self.fusion_activation = fus_act_name
        cls_act = build_activation(cls_activation)
        cme_act = build_activation(cme_activation)
        fus_act = build_activation(fus_act_name)

        # Gene embedder: G → [gene_hidden]* → dx (depth configurable)
        self.mlp_in_node_features = _build_gene_mlp(
            gene_dim,
            self.gene_hidden_dim,
            self.dx,
            self.gene_mlp_depth,
            gene_act_names,
        )

        self.transformer_layers = nn.ModuleList(
            [
                GeneTransformerLayer(
                    node_features_dimensions=self.dx,
                    num_heads=int(hidden_dims["num_heads"]),
                    dim_ff_node_features=int(hidden_dims["dim_ffX"]),
                    dropout=dropout_layer,
                    ffn_activation=layer_activation,
                    attention_use_node_mask=self.attention_use_node_mask,
                    pre_norm=self.pre_norm,
                )
                for _ in range(n_layers)
            ]
        )

        head_in = self._head_input_dim()
        self.head_input_dim = head_in

        # Parallel heads (no cross) — kept when feature_cross is off.
        self.cls_head = None
        self.cme_head = None
        self.presence_head = None
        self.comp_head = None
        # Cross pathway modules (None when feature_cross is off).
        self.type_stage1 = None
        self.env_stage1 = None
        self.type_probe = None
        self.env_probe = None
        self.fusion_mlp = None
        self.type_stage2 = None
        self.env_stage2 = None
        self.presence_stage2 = None
        self.comp_stage2 = None

        if not self.feature_cross:
            self.cls_head = _mlp_head(
                head_in,
                self.cls_hidden_dim,
                num_classes,
                build_activation(cls_activation),
                dropout_cls,
            )
            if self.predict_cme:
                if self.cme_presence_gate:
                    self.presence_head = _mlp_head(
                        head_in,
                        self.presence_hidden_dim,
                        num_classes,
                        build_activation(cme_activation),
                        dropout_cme,
                    )
                    self.comp_head = _mlp_head(
                        head_in,
                        self.cme_hidden_dim,
                        num_classes,
                        build_activation(cme_activation),
                        dropout_cme,
                    )
                else:
                    self.cme_head = _mlp_head(
                        head_in,
                        self.cme_hidden_dim,
                        num_classes,
                        build_activation(cme_activation),
                        dropout_cme,
                    )
        else:
            # Intermediate dims for stage-1 towers.
            self.type_stage1_dim = int(
                hidden_mlp_dims.get("type_stage1", self.cls_hidden_dim)
            )
            self.env_stage1_dim = int(
                hidden_mlp_dims.get("env_stage1", self.cme_hidden_dim)
            )
            fusion_hidden = int(
                hidden_mlp_dims.get(
                    "fusion", max(self.type_stage1_dim, self.env_stage1_dim)
                )
            )
            # Fuse back to trunk width (head_in) before stage-2.
            self.fusion_out_dim = head_in
            fusion_in = self.type_stage1_dim + self.env_stage1_dim
            if self.feature_cross_include_h:
                fusion_in += head_in

            self.type_stage1 = _stage1_tower(
                head_in,
                self.type_stage1_dim,
                build_activation(cls_activation),
                dropout_cls,
            )
            self.env_stage1 = _stage1_tower(
                head_in,
                self.env_stage1_dim,
                build_activation(cme_activation),
                dropout_cme,
            )
            # Free-standing probes from stage-1 features (ablation / monitoring).
            self.type_probe = nn.Linear(self.type_stage1_dim, num_classes)
            self.env_probe = nn.Linear(self.env_stage1_dim, num_classes)

            self.fusion_mlp = nn.Sequential(
                nn.LayerNorm(fusion_in),
                nn.Linear(fusion_in, fusion_hidden),
                fus_act,
                nn.Dropout(dropout_fusion),
                nn.Linear(fusion_hidden, self.fusion_out_dim),
            )
            self.fusion_mlp_cme = None
            if self.dual_fusion:
                self.fusion_mlp_cme = nn.Sequential(
                    nn.LayerNorm(fusion_in),
                    nn.Linear(fusion_in, fusion_hidden),
                    build_activation(fus_act_name),
                    nn.Dropout(dropout_fusion),
                    nn.Linear(fusion_hidden, self.fusion_out_dim),
                )
            self.type_stage2 = _mlp_head(
                self.fusion_out_dim,
                self.cls_hidden_dim,
                num_classes,
                build_activation(cls_activation),
                dropout_cls,
            )
            cme_in = self.fusion_out_dim + (
                num_classes if self.cme_condition_on_cls else 0
            )
            if self.cme_presence_gate:
                self.presence_stage2 = _mlp_head(
                    cme_in,
                    self.presence_hidden_dim,
                    num_classes,
                    build_activation(cme_activation),
                    dropout_cme,
                )
                self.comp_stage2 = _mlp_head(
                    cme_in,
                    self.cme_hidden_dim,
                    num_classes,
                    build_activation(cme_activation),
                    dropout_cme,
                )
                # Keep env_stage2 as an alias of composition for stage-1-style
                # tooling that still looks for a single CME stage-2 module.
                self.env_stage2 = self.comp_stage2
            else:
                self.env_stage2 = _mlp_head(
                    cme_in,
                    self.cme_hidden_dim,
                    num_classes,
                    build_activation(cme_activation),
                    dropout_cme,
                )

        self._init_presence_bias()

    def _init_presence_bias(self) -> None:
        """Optionally bias presence logits so π starts near-saturated (gate≈id)."""
        if not self.cme_presence_gate or abs(self.presence_bias_init) < 1e-12:
            return
        for head in (self.presence_head, self.presence_stage2):
            if head is None:
                continue
            # Last Linear in Sequential MLP head.
            last = None
            for mod in head.modules():
                if isinstance(mod, nn.Linear):
                    last = mod
            if last is not None and last.bias is not None:
                nn.init.constant_(last.bias, float(self.presence_bias_init))

    @staticmethod
    def compose_gated_cme(
        presence_logits: torch.Tensor,
        comp_logits: torch.Tensor,
        *,
        eps: float = 1e-8,
        presence_topk: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """π = σ(presence), q = softmax(comp), p = normalize(π ⊙ q).

        Returns ``(presence_probs, comp_probs, cme_probs)``.
        If ``presence_topk > 0``, zero all but the top-k presence values per
        cell before the product (hard sparse support).
        """
        presence = torch.sigmoid(presence_logits)
        k = int(presence_topk)
        if k > 0:
            k = min(k, presence.size(-1))
            topv, topi = torch.topk(presence, k, dim=-1)
            mask = torch.zeros_like(presence)
            mask.scatter_(-1, topi, 1.0)
            presence = presence * mask
        comp = F.softmax(comp_logits, dim=-1)
        gated = presence * comp
        cme_probs = gated / gated.sum(dim=-1, keepdim=True).clamp_min(eps)
        return presence, comp, cme_probs

    def _head_input_dim(self) -> int:
        """Channel count into task heads given skip / summary options."""
        if self.global_skip == "concat":
            dim = 2 * self.dx
        else:
            # ``add`` or disabled: cell stream stays ``dx``.
            dim = self.dx
        if self.subgraph_summary:
            if self.subgraph_pool == "mean_max":
                dim += 2 * self.dx
            else:
                dim += self.dx
        if self.gene_knn_k > 0:
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
            pool_src = e_raw if self.subgraph_pool_source == "raw" else e_mixed
            if self.subgraph_pool == "attn":
                assert self.subgraph_attn_pool is not None
                summary = self.subgraph_attn_pool(pool_src, node_mask)
            elif self.subgraph_pool == "mean_max":
                mean_summary = masked_mean_pool(pool_src, node_mask)
                max_summary = masked_max_pool(pool_src, node_mask)
                summary = torch.cat([mean_summary, max_summary], dim=-1)
            else:
                summary = masked_mean_pool(pool_src, node_mask)
            summary = summary.expand(-1, cell_feat.size(1), -1)
            cell_feat = torch.cat([cell_feat, summary], dim=-1)

        if self.gene_knn_k > 0:
            knn_summary = gene_knn_mean_pool(
                e_raw, e_mixed, node_mask, k=self.gene_knn_k
            )
            cell_feat = torch.cat([cell_feat, knn_summary], dim=-1)

        cell_feat = cell_feat * node_mask.unsqueeze(-1).to(cell_feat.dtype)
        return cell_feat

    def encode(self, data: DataHolder) -> torch.Tensor:
        """Return features fed to the classifier ``(B, N, head_input_dim)``."""
        e_raw, e_mixed = self.encode_raw_and_mixed(data)
        return self.fuse_for_head(e_raw, e_mixed, data.node_mask)

    def forward(self, data: DataHolder) -> dict:
        """
        Args:
            data: ``node_features`` ``(B, N, G)``, ``node_mask`` ``(B, N)``.

        Returns:
            dict with:
              * ``cls_logits`` / ``cme_logits`` — final scores used for training
              * if ``cme_presence_gate``: also ``presence_logits``, ``comp_logits``,
                ``cme_probs`` (composed mix; prefer this over softmaxing logits)
              * if ``feature_cross``: also ``cls_logits_stage1``,
                ``cme_logits_stage1`` (probes from stage-1 features)
        """
        h = self.encode(data)
        mask = data.node_mask.unsqueeze(-1).to(h.dtype)

        if not self.feature_cross:
            out = {"cls_logits": self.cls_head(h) * mask}
            if self.predict_cme:
                if self.cme_presence_gate:
                    presence_logits = self.presence_head(h) * mask
                    comp_logits = self.comp_head(h) * mask
                    _, _, cme_probs = self.compose_gated_cme(
                        presence_logits,
                        comp_logits,
                        presence_topk=self.presence_topk,
                    )
                    cme_probs = cme_probs * mask
                    out.update(
                        {
                            "presence_logits": presence_logits,
                            "comp_logits": comp_logits,
                            "cme_probs": cme_probs,
                            # Log-space probs for SoftCE paths that still
                            # apply log_softmax: use ``cme_probs`` in loss.
                            "cme_logits": torch.log(cme_probs.clamp_min(1e-8))
                            * mask,
                        }
                    )
                elif self.cme_head is not None:
                    out["cme_logits"] = self.cme_head(h) * mask
            return out

        t = self.type_stage1(h)
        e = self.env_stage1(h)
        if self.feature_cross_include_h:
            fused_in = torch.cat([t, e, h], dim=-1)
        else:
            fused_in = torch.cat([t, e], dim=-1)
        f = self.fusion_mlp(fused_in)
        f = f * mask  # pads stay zero into stage-2

        if self.dual_fusion and self.fusion_mlp_cme is not None:
            f_cme = self.fusion_mlp_cme(fused_in) * mask
        elif self.cme_detach_type_in_fusion:
            t_cme = t.detach()
            if self.feature_cross_include_h:
                fused_in_cme = torch.cat([t_cme, e, h], dim=-1)
            else:
                fused_in_cme = torch.cat([t_cme, e], dim=-1)
            f_cme = self.fusion_mlp(fused_in_cme) * mask
        else:
            f_cme = f

        cls_logits = self.type_stage2(f) * mask
        if self.cme_condition_on_cls:
            # Detached self-type signal for type-conditional microenvironment.
            t_probs = F.softmax(self.type_probe(t).detach(), dim=-1)
            f_cme_in = torch.cat([f_cme, t_probs], dim=-1)
        else:
            f_cme_in = f_cme

        out = {
            "cls_logits": cls_logits,
            "cls_logits_stage1": self.type_probe(t) * mask,
            "cme_logits_stage1": self.env_probe(e) * mask,
        }

        if self.cme_presence_gate:
            presence_logits = self.presence_stage2(f_cme_in) * mask
            comp_logits = self.comp_stage2(f_cme_in) * mask
            _, _, cme_probs = self.compose_gated_cme(
                presence_logits,
                comp_logits,
                presence_topk=self.presence_topk,
            )
            cme_probs = cme_probs * mask
            cme_logits = torch.log(cme_probs.clamp_min(1e-8)) * mask
            out.update(
                {
                    "presence_logits": presence_logits,
                    "comp_logits": comp_logits,
                    "cme_probs": cme_probs,
                    "cme_logits": cme_logits,
                }
            )
        else:
            cme_logits = self.env_stage2(f_cme_in) * mask
            if abs(self.cme_entropy_temp_alpha) > 1e-12 or abs(
                self.cme_entropy_temp_t0 - 1.0
            ) > 1e-12:
                # Per-cell temperature from predictive entropy (eval/calib).
                logp = F.log_softmax(cme_logits, dim=-1)
                p = logp.exp()
                H = -(p * logp).sum(dim=-1, keepdim=True)
                T = (
                    self.cme_entropy_temp_t0
                    + self.cme_entropy_temp_alpha * (H - H.mean())
                ).clamp(0.7, 1.4)
                cme_logits = (cme_logits / T) * mask
            out["cme_logits"] = cme_logits
        return out
