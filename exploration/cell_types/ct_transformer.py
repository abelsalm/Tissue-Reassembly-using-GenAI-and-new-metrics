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
      * ``subgraph_summary``: if True, masked mean-pool of ``E_mixed`` is
        broadcast and concatenated as a bag-level neighborhood descriptor.
      * ``feature_cross``: if True (requires ``predict_cme``), stage-1 type/env
        towers produce intermediates ``t_i``, ``e_i``; fuse with optional ``h_i``
        via LN+MLP; stage-2 heads read the fused features for final logits.
        Stage-1 linear probes are also returned for ablation.

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
        cls_activation: str = "relu",
        cme_activation: str = "relu",
        fusion_activation: Optional[str] = None,
        global_skip: Optional[str] = None,
        subgraph_summary: bool = False,
        predict_cme: bool = False,
        feature_cross: bool = False,
        feature_cross_include_h: bool = True,
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
        self.global_skip = _normalize_global_skip(global_skip)
        self.subgraph_summary = bool(subgraph_summary)
        self.predict_cme = bool(predict_cme)
        self.feature_cross = bool(feature_cross)
        self.feature_cross_include_h = bool(feature_cross_include_h)

        if self.feature_cross and not self.predict_cme:
            raise ValueError("feature_cross=True requires predict_cme=True")

        act1_name, act2_name = _resolve_gene_mlp_activations(gene_mlp_activation)
        self.gene_mlp_activation = (act1_name, act2_name)
        self.cls_activation = str(cls_activation)
        self.cme_activation = str(cme_activation)
        fus_act_name = (
            str(fusion_activation)
            if fusion_activation is not None
            else str(cls_activation)
        )
        self.fusion_activation = fus_act_name
        act1 = build_activation(act1_name)
        act2 = build_activation(act2_name)
        cls_act = build_activation(cls_activation)
        cme_act = build_activation(cme_activation)
        fus_act = build_activation(fus_act_name)

        # Gene embedder: G → gene_hidden → dx
        self.mlp_in_node_features = nn.Sequential(
            nn.Linear(gene_dim, self.gene_hidden_dim),
            act1,
            nn.Linear(self.gene_hidden_dim, self.dx),
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

        # Parallel heads (no cross) — kept when feature_cross is off.
        self.cls_head = None
        self.cme_head = None
        # Cross pathway modules (None when feature_cross is off).
        self.type_stage1 = None
        self.env_stage1 = None
        self.type_probe = None
        self.env_probe = None
        self.fusion_mlp = None
        self.type_stage2 = None
        self.env_stage2 = None

        if not self.feature_cross:
            self.cls_head = _mlp_head(
                head_in,
                self.cls_hidden_dim,
                num_classes,
                build_activation(cls_activation),
                dropout_cls,
            )
            if self.predict_cme:
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
            self.type_stage2 = _mlp_head(
                self.fusion_out_dim,
                self.cls_hidden_dim,
                num_classes,
                build_activation(cls_activation),
                dropout_cls,
            )
            self.env_stage2 = _mlp_head(
                self.fusion_out_dim,
                self.cme_hidden_dim,
                num_classes,
                build_activation(cme_activation),
                dropout_cme,
            )

    def _head_input_dim(self) -> int:
        """Channel count into task heads given skip / summary options."""
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

    def forward(self, data: DataHolder) -> dict:
        """
        Args:
            data: ``node_features`` ``(B, N, G)``, ``node_mask`` ``(B, N)``.

        Returns:
            dict with:
              * ``cls_logits`` / ``cme_logits`` — final scores used for training
              * if ``feature_cross``: also ``cls_logits_stage1``,
                ``cme_logits_stage1`` (probes from stage-1 features)
        """
        h = self.encode(data)
        mask = data.node_mask.unsqueeze(-1).to(h.dtype)

        if not self.feature_cross:
            out = {"cls_logits": self.cls_head(h) * mask}
            if self.predict_cme and self.cme_head is not None:
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

        out = {
            "cls_logits": self.type_stage2(f) * mask,
            "cme_logits": self.env_stage2(f) * mask,
            "cls_logits_stage1": self.type_probe(t) * mask,
            "cme_logits_stage1": self.env_probe(e) * mask,
        }
        return out
