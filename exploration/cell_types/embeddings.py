"""Learnable lookup embeddings for cell-level categorical inputs.

``CellTypeEmbedding`` is a classical ``nn.Embedding`` of dimension ``n``.
Each class row is initialized to the **mean of that class in the space of
the first ``n`` principal components** of gene expression (global PCA,
then per-class mean of the PC scores). Index 0 is reserved for UNK / pad
when ``reserve_unk=True`` (same convention as ``CellTypeTransformer``).

``CellTypeMDSLoss`` regularizes the table so pairwise distances between
class embeddings stay close to pairwise distances between class means in
**real transcriptomic space** (metric MDS / Kruskal-style stress).

``DomainEmbedding`` is an **amortized** encoder, not a lookup table. For
each input graph / slice it maps every domain to the pooled PCA
transcriptome of cells in that domain (mean, or attention pooling).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

MDS_SCALES = ("mean", "none", "optimal")
MDS_WEIGHTINGS = ("uniform", "sammon")
DOMAIN_POOL_MODES = ("mean", "attention")


class CellTypeEmbedding(nn.Module):
    """Cell-type lookup table of size ``n``, PCA-mean initialized.

    Parameters
    ----------
    num_classes:
        Number of real cell-type classes (``C``). With ``reserve_unk=True``
        the table has ``C + 1`` rows and class ``k`` lives at index ``k + 1``.
    n:
        Embedding dimension, equal to the number of leading PCs used to
        initialize each class vector.
    reserve_unk:
        If True (default), index 0 is UNK / pad (zeros, ``padding_idx=0``).
    freeze:
        If True, the table is not trained.
    """

    def __init__(
        self,
        num_classes: int,
        n: int,
        *,
        reserve_unk: bool = True,
        freeze: bool = False,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1 (got {num_classes})")
        if n < 1:
            raise ValueError(f"n (embedding dim / n PCs) must be >= 1 (got {n})")
        self.num_classes = int(num_classes)
        self.n = int(n)
        self.reserve_unk = bool(reserve_unk)
        self.class_names: List[str] = (
            [str(x) for x in class_names] if class_names is not None else []
        )
        num_embeddings = self.num_classes + (1 if self.reserve_unk else 0)
        padding_idx = 0 if self.reserve_unk else None
        self.embed = nn.Embedding(
            num_embeddings, self.n, padding_idx=padding_idx
        )
        if freeze:
            self.embed.weight.requires_grad_(False)
        self.mds: Optional[CellTypeMDSLoss] = None

    @property
    def num_embeddings(self) -> int:
        return int(self.embed.num_embeddings)

    def class_rows(self) -> torch.Tensor:
        """Trainable class vectors ``(C, n)``, excluding the UNK row."""
        weight = self.embed.weight
        if self.reserve_unk:
            return weight[1 : 1 + self.num_classes]
        return weight

    def forward(self, type_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings. ``type_ids`` is any integer tensor of class ids."""
        ids = type_ids.long()
        if ids.dim() > 0 and ids.size(-1) == 1 and ids.dim() >= 2:
            ids = ids.squeeze(-1)
        return self.embed(ids.clamp(min=0, max=self.num_embeddings - 1))

    @torch.no_grad()
    def init_from_mean_pca(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        standardize: bool = True,
        pca_max_cells: Optional[int] = None,
        build_mds: bool = True,
    ) -> "CellTypeEmbedding":
        """Fill the table with per-class mean PCA vectors of dimension ``n``.

        ``features`` is ``(N, G)`` gene expression, ``labels`` is ``(N,)``
        with integer ids in ``0 … C-1``. PCA is fit globally; each class
        row is the mean of that class after projection onto the first
        ``n`` components.
        """
        matrix = class_mean_pca_matrix(
            features,
            labels,
            n_components=self.n,
            num_classes=self.num_classes,
            standardize=standardize,
            pca_max_cells=pca_max_cells,
        )
        weight = self.embed.weight
        if self.reserve_unk:
            weight.zero_()
            weight[1 : 1 + self.num_classes].copy_(
                matrix.to(device=weight.device, dtype=weight.dtype)
            )
        else:
            weight.copy_(matrix.to(device=weight.device, dtype=weight.dtype))
        if build_mds:
            self.mds = CellTypeMDSLoss.from_features(
                features,
                labels,
                num_classes=self.num_classes,
                standardize=standardize,
            ).to(device=weight.device)
        else:
            self.mds = None
        return self

    def mds_loss(
        self,
        *,
        scale: str = "mean",
        weighting: str = "uniform",
    ) -> torch.Tensor:
        """MDS regularizer: embedding distances vs transcriptomic class distances.

        Requires a prior ``init_from_mean_pca`` / ``from_features`` /
        ``from_dataset`` call so the target pairwise geometry is cached.
        """
        if self.mds is None:
            return self.embed.weight.new_zeros(())
        return self.mds(
            self.class_rows(), scale=scale, weighting=weighting
        )

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        labels: torch.Tensor,
        n: int,
        *,
        num_classes: Optional[int] = None,
        reserve_unk: bool = True,
        freeze: bool = False,
        class_names: Optional[Sequence[str]] = None,
        standardize: bool = True,
        pca_max_cells: Optional[int] = None,
        build_mds: bool = True,
    ) -> "CellTypeEmbedding":
        """Build a ``CellTypeEmbedding`` and initialize it from gene PCA means."""
        labels_1d = _as_label_ids(labels)
        if num_classes is None:
            num_classes = int(labels_1d.max().item()) + 1
        module = cls(
            num_classes=int(num_classes),
            n=int(n),
            reserve_unk=reserve_unk,
            freeze=freeze,
            class_names=class_names,
        )
        module.init_from_mean_pca(
            features,
            labels_1d,
            standardize=standardize,
            pca_max_cells=pca_max_cells,
            build_mds=build_mds,
        )
        return module

    @classmethod
    def from_dataset(
        cls,
        dataset,
        n: int,
        *,
        reserve_unk: bool = True,
        freeze: bool = False,
        standardize: bool = True,
        pca_max_cells: Optional[int] = None,
        build_mds: bool = True,
    ) -> "CellTypeEmbedding":
        """Initialize from a processed ``Dataset`` (``_data.node_features``)."""
        data = dataset._data
        decoder = getattr(getattr(dataset, "statistics", None), "cell_class_decoder", None)
        names: Optional[List[str]] = None
        if isinstance(decoder, dict) and decoder:
            names = [str(decoder[i]) for i in range(len(decoder))]
        num_classes = int(getattr(dataset, "num_cell_class", 0) or 0)
        if num_classes < 1:
            num_classes = None
        return cls.from_features(
            data.node_features,
            data.cell_class,
            n=n,
            num_classes=num_classes,
            reserve_unk=reserve_unk,
            freeze=freeze,
            class_names=names,
            standardize=standardize,
            pca_max_cells=pca_max_cells,
            build_mds=build_mds,
        )


class DomainEmbedding(nn.Module):
    """Amortized domain embedding from a graph's own transcriptomes.

    Unlike ``CellTypeEmbedding``, there is no global lookup row per domain.
    For each input graph / slice, cells are projected with a shared PCA of
    size ``n``, then **pooled per domain**:

    * ``pool="mean"`` — average PCA vectors of cells in that domain.
    * ``pool="attention"`` — gated attention over those cells; scores come
      from a learned function of the ``n`` PCA coordinates (the variables
      in each vector).

    ``variable_attention=True`` adds a second softmax over the ``n``
    coordinates of the pooled vector, so PCA axes can be reweighted.

    Forward returns a per-cell tensor ``(B, N, n)``: every cell receives
    the embedding of its domain in that graph (zeros for pad / UNK).
    """

    def __init__(
        self,
        num_domains: int,
        n: int,
        gene_dim: int,
        *,
        pool: str = "mean",
        variable_attention: bool = False,
        reserve_unk: bool = False,
        freeze_pca: bool = True,
        attn_hidden: Optional[int] = None,
        standardize: bool = True,
    ) -> None:
        super().__init__()
        if num_domains < 1:
            raise ValueError(f"num_domains must be >= 1 (got {num_domains})")
        if n < 1:
            raise ValueError(f"n (embedding dim / n PCs) must be >= 1 (got {n})")
        if gene_dim < 1:
            raise ValueError(f"gene_dim must be >= 1 (got {gene_dim})")
        self.num_domains = int(num_domains)
        self.n = int(n)
        self.gene_dim = int(gene_dim)
        self.pool = _validate_choice(pool, DOMAIN_POOL_MODES, "pool")
        self.variable_attention = bool(variable_attention)
        self.reserve_unk = bool(reserve_unk)
        self.freeze_pca = bool(freeze_pca)
        self.standardize = bool(standardize)
        self.register_buffer("pca_mean", torch.zeros(self.gene_dim))
        self.register_buffer("pca_scale", torch.ones(self.gene_dim))
        self.proj = nn.Linear(self.gene_dim, self.n, bias=False)
        nn.init.zeros_(self.proj.weight)
        if self.freeze_pca:
            self.proj.weight.requires_grad_(False)
        self.cell_attn: Optional[nn.Module] = None
        if self.pool == "attention":
            hidden = int(attn_hidden) if attn_hidden is not None else self.n
            if hidden < 1:
                raise ValueError(f"attn_hidden must be >= 1 (got {hidden})")
            self.cell_attn = nn.Sequential(
                nn.Linear(self.n, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1, bias=False),
            )
        self.var_attn: Optional[nn.Linear] = None
        if self.variable_attention:
            self.var_attn = nn.Linear(self.n, self.n)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Map gene expression ``(..., G)`` to PCA scores ``(..., n)``."""
        scale = self.pca_scale.clamp_min(1e-6)
        z = (features - self.pca_mean) / scale
        return self.proj(z)

    @torch.no_grad()
    def fit_pca(
        self,
        features: torch.Tensor,
        *,
        pca_max_cells: Optional[int] = None,
        standardize: Optional[bool] = None,
    ) -> "DomainEmbedding":
        """Fit the shared PCA projector on ``(N, G)`` gene expression."""
        x = features.detach()
        if x.dim() != 2:
            raise ValueError(f"features must be (N, G), got {tuple(x.shape)}")
        if x.size(1) != self.gene_dim:
            raise ValueError(
                f"features gene dim {x.size(1)} != gene_dim={self.gene_dim}"
            )
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        do_std = self.standardize if standardize is None else bool(standardize)
        pca_x = _subsample_rows(x, pca_max_cells)
        mean, scale, components = _fit_pca(
            pca_x, n_components=self.n, standardize=do_std
        )
        self.pca_mean.copy_(
            mean.to(device=self.pca_mean.device, dtype=self.pca_mean.dtype)
        )
        if scale is None:
            self.pca_scale.fill_(1.0)
        else:
            self.pca_scale.copy_(
                scale.to(device=self.pca_scale.device, dtype=self.pca_scale.dtype)
            )
        weight = self.proj.weight
        weight.zero_()
        n_kept = int(components.size(1))
        weight[:n_kept].copy_(
            components.T.to(device=weight.device, dtype=weight.dtype)
        )
        return self

    def forward(
        self,
        features: torch.Tensor,
        domain_ids: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        *,
        return_table: bool = False,
    ):
        """Pool per-domain PCA vectors inside each graph.

        Parameters
        ----------
        features:
            ``(B, N, G)`` or ``(N, G)`` gene expression.
        domain_ids:
            Integer domain ids, same leading shape as ``features``.
        node_mask:
            Optional ``(B, N)`` / ``(N,)`` mask (1 = real cell).
        return_table:
            If True, also return the ``(B, D, n)`` per-graph domain table.
        """
        features, domain_ids, node_mask, squeezed = _as_graph_inputs(
            features, domain_ids, node_mask
        )
        if features.size(-1) != self.gene_dim:
            raise ValueError(
                f"features gene dim {features.size(-1)} != gene_dim={self.gene_dim}"
            )
        z = self.project(features)
        out, table = self._pool_domains(z, domain_ids, node_mask)
        if squeezed:
            out = out.squeeze(0)
            table = table.squeeze(0)
        if return_table:
            return out, table
        return out

    def _pool_domains(
        self,
        z: torch.Tensor,
        domain_ids: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_cells, n_dim = z.shape
        n_domains = self.num_domains
        ids = domain_ids.long()
        valid = node_mask.bool()
        if self.reserve_unk:
            valid = valid & (ids > 0) & (ids < n_domains)
        else:
            valid = valid & (ids >= 0) & (ids < n_domains)
        ids = ids.clamp(0, n_domains - 1)
        batch_index = torch.arange(batch_size, device=z.device).unsqueeze(1).expand(
            batch_size, n_cells
        )
        keys = batch_index * n_domains + ids
        table_rows = batch_size * n_domains
        z_flat = z.reshape(batch_size * n_cells, n_dim)
        keys_flat = keys.reshape(batch_size * n_cells)
        valid_flat = valid.reshape(batch_size * n_cells)
        src = z_flat[valid_flat]
        src_keys = keys_flat[valid_flat]
        if self.pool == "attention" and self.cell_attn is not None:
            scores = self.cell_attn(z).squeeze(-1).reshape(batch_size * n_cells)
            table = _scatter_attn_pool(src, scores[valid_flat], src_keys, table_rows)
        else:
            table = _scatter_mean(src, src_keys, table_rows)
        table = table.view(batch_size, n_domains, n_dim)
        if self.variable_attention and self.var_attn is not None:
            gates = torch.softmax(self.var_attn(table), dim=-1)
            table = table * gates * float(self.n)
        out = table.gather(
            1, ids.unsqueeze(-1).expand(batch_size, n_cells, n_dim)
        )
        out = out * valid.unsqueeze(-1).to(dtype=out.dtype)
        return out, table

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        n: int,
        num_domains: int,
        *,
        pool: str = "mean",
        variable_attention: bool = False,
        reserve_unk: bool = False,
        freeze_pca: bool = True,
        attn_hidden: Optional[int] = None,
        standardize: bool = True,
        pca_max_cells: Optional[int] = None,
    ) -> "DomainEmbedding":
        """Build a ``DomainEmbedding`` and fit PCA on ``(N, G)`` genes."""
        if features.dim() != 2:
            raise ValueError(f"features must be (N, G), got {tuple(features.shape)}")
        module = cls(
            num_domains=int(num_domains),
            n=int(n),
            gene_dim=int(features.size(1)),
            pool=pool,
            variable_attention=variable_attention,
            reserve_unk=reserve_unk,
            freeze_pca=freeze_pca,
            attn_hidden=attn_hidden,
            standardize=standardize,
        )
        module.fit_pca(features, pca_max_cells=pca_max_cells, standardize=standardize)
        return module

    @classmethod
    def from_dataset(
        cls,
        dataset,
        n: int,
        *,
        num_domains: Optional[int] = None,
        pool: str = "mean",
        variable_attention: bool = False,
        reserve_unk: bool = False,
        freeze_pca: bool = True,
        attn_hidden: Optional[int] = None,
        standardize: bool = True,
        pca_max_cells: Optional[int] = None,
    ) -> "DomainEmbedding":
        """Fit PCA from a processed ``Dataset`` (``_data.node_features``)."""
        data = dataset._data
        if num_domains is None:
            num_domains = int(getattr(dataset, "num_cell_class", 0) or 0)
        if num_domains < 1:
            raise ValueError(
                "num_domains must be given or available on the dataset"
            )
        return cls.from_features(
            data.node_features,
            n=n,
            num_domains=int(num_domains),
            pool=pool,
            variable_attention=variable_attention,
            reserve_unk=reserve_unk,
            freeze_pca=freeze_pca,
            attn_hidden=attn_hidden,
            standardize=standardize,
            pca_max_cells=pca_max_cells,
        )


class CellTypeMDSLoss(nn.Module):
    """Metric MDS regularizer: keep class embeddings isometric to transcriptome.

    Target distances are pairwise Euclidean distances between **per-class
    mean gene-expression vectors** (z-scored genes by default). The loss
    compares those to pairwise distances of the current embedding rows
    (UNK / empty classes are dropped).

    Scale
    -----
    * ``mean`` (default) — divide each upper-triangle distance vector by
      its mean, then MSE. Matches relative geometry when embedding units
      differ from transcriptomic units.
    * ``none`` — raw MSE of distances (LUNA-style pairwise distance MSE).
    * ``optimal`` — ratio MDS: closed-form ``α`` with ``α · d_embed ≈ d_tx``.

    Weighting
    ---------
    * ``uniform`` — mean over occupied class pairs.
    * ``sammon`` — Sammon stress, pairs weighted by ``1 / d_tx``.
    """

    def __init__(
        self,
        target_dist: torch.Tensor,
        occupied: torch.Tensor,
        *,
        scale: str = "mean",
        weighting: str = "uniform",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if target_dist.dim() != 2 or target_dist.size(0) != target_dist.size(1):
            raise ValueError(
                "target_dist must be a square (C, C) matrix, "
                f"got {tuple(target_dist.shape)}"
            )
        if occupied.shape != (target_dist.size(0),):
            raise ValueError(
                f"occupied must be ({target_dist.size(0)},), "
                f"got {tuple(occupied.shape)}"
            )
        self.scale = _validate_choice(scale, MDS_SCALES, "scale")
        self.weighting = _validate_choice(weighting, MDS_WEIGHTINGS, "weighting")
        self.eps = float(eps)
        self.register_buffer("target_dist", target_dist.detach().float())
        self.register_buffer("occupied", occupied.detach().bool())

    @classmethod
    def from_features(
        cls,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        num_classes: Optional[int] = None,
        standardize: bool = True,
        scale: str = "mean",
        weighting: str = "uniform",
        eps: float = 1e-8,
    ) -> "CellTypeMDSLoss":
        """Cache class-mean pairwise distances in transcriptomic space."""
        dist, occupied = transcriptome_pair_distances(
            features,
            labels,
            num_classes=num_classes,
            standardize=standardize,
        )
        return cls(
            target_dist=dist,
            occupied=occupied,
            scale=scale,
            weighting=weighting,
            eps=eps,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        scale: Optional[str] = None,
        weighting: Optional[str] = None,
    ) -> torch.Tensor:
        """``embeddings`` is ``(C, n)`` class rows in the same class order."""
        if embeddings.dim() != 2:
            raise ValueError(
                f"embeddings must be (C, n), got {tuple(embeddings.shape)}"
            )
        n_classes = int(self.target_dist.size(0))
        if embeddings.size(0) != n_classes:
            raise ValueError(
                f"embeddings has {embeddings.size(0)} rows, "
                f"expected {n_classes} classes"
            )
        d_embed = pairwise_euclidean(embeddings)
        d_target = self.target_dist.to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        occupied = self.occupied.to(device=embeddings.device)
        return mds_stress(
            d_embed,
            d_target,
            occupied,
            scale=self.scale if scale is None else scale,
            weighting=self.weighting if weighting is None else weighting,
            eps=self.eps,
        )


def transcriptome_pair_distances(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: Optional[int] = None,
    standardize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pairwise Euclidean distances of class means in transcriptomic space.

    Returns ``(dist, occupied)`` with ``dist`` of shape ``(C, C)`` and
    ``occupied`` of shape ``(C,)``. Genes are z-scored over cells when
    ``standardize`` is True.
    """
    x = features.detach()
    if x.dim() != 2:
        raise ValueError(f"features must be (N, G), got {tuple(x.shape)}")
    y = _as_label_ids(labels)
    if y.numel() != x.size(0):
        raise ValueError(
            f"labels length {y.numel()} does not match N={x.size(0)}"
        )
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if num_classes is None:
        num_classes = int(y.max().item()) + 1 if y.numel() else 0
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    class_means, occupied = _class_means(x, y, num_classes)
    if standardize:
        # Standardization is affine, so standardizing class means is exactly
        # equivalent to standardizing every cell before averaging, without
        # allocating another potentially multi-GB (N, G) tensor.
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        class_means = (class_means - mean) / std
        class_means[~occupied] = 0
    dist = pairwise_euclidean(class_means)
    dist = dist.masked_fill(~occupied.unsqueeze(0), 0.0)
    dist = dist.masked_fill(~occupied.unsqueeze(1), 0.0)
    return dist, occupied


def pairwise_euclidean(points: torch.Tensor) -> torch.Tensor:
    """Pairwise L2 distances for a point set ``(C, D)`` → ``(C, C)``."""
    if points.size(0) == 0:
        return points.new_zeros(0, 0)
    return torch.cdist(points, points, p=2)


def mds_stress(
    d_embed: torch.Tensor,
    d_target: torch.Tensor,
    occupied: torch.Tensor,
    *,
    scale: str = "mean",
    weighting: str = "uniform",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean MDS stress over occupied class pairs ``i < j``.

    ``d_embed`` / ``d_target`` are ``(C, C)`` distance matrices.
    """
    scale_key = _validate_choice(scale, MDS_SCALES, "scale")
    weight_key = _validate_choice(weighting, MDS_WEIGHTINGS, "weighting")
    pair_mask = _upper_occupied_pairs(occupied)
    if int(pair_mask.sum().item()) < 1:
        return d_embed.new_zeros(())

    de = d_embed[pair_mask]
    dt = d_target[pair_mask]
    floor = de.new_tensor(float(eps))
    if scale_key == "mean":
        de = de / de.mean().clamp_min(floor)
        dt = dt / dt.mean().clamp_min(floor)
    elif scale_key == "optimal":
        alpha = (de * dt).sum() / de.pow(2).sum().clamp_min(floor)
        de = alpha * de
    delta2 = (de - dt).pow(2)
    if weight_key == "uniform":
        return delta2.mean()
    weights = dt.clamp_min(floor).reciprocal()
    return (weights * delta2).sum() / weights.sum().clamp_min(floor)


def _upper_occupied_pairs(occupied: torch.Tensor) -> torch.Tensor:
    n = int(occupied.numel())
    iu = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=occupied.device),
        diagonal=1,
    )
    both = occupied.unsqueeze(0) & occupied.unsqueeze(1)
    return iu & both


def _validate_choice(value: str, allowed: Sequence[str], name: str) -> str:
    key = str(value).strip().lower()
    if key not in allowed:
        raise ValueError(
            f"{name} must be one of {list(allowed)} (got {value!r})"
        )
    return key


def class_mean_pca_matrix(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_components: int,
    *,
    num_classes: Optional[int] = None,
    standardize: bool = True,
    pca_max_cells: Optional[int] = None,
) -> torch.Tensor:
    """Per-class mean of the first ``n_components`` global PCA scores.

    Returns a float tensor of shape ``(C, n)``. Empty classes are zeros.
    PCA is fit on (optionally subsampled) cells; class means use every cell.
    """
    x = features.detach()
    if x.dim() != 2:
        raise ValueError(f"features must be (N, G), got {tuple(x.shape)}")
    y = _as_label_ids(labels)
    if y.numel() != x.size(0):
        raise ValueError(
            f"labels length {y.numel()} does not match N={x.size(0)}"
        )
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    n_cells, n_genes = int(x.size(0)), int(x.size(1))
    if num_classes is None:
        num_classes = int(y.max().item()) + 1 if n_cells else 0
    num_classes = int(num_classes)
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    class_means, occupied = _class_means(x, y, num_classes)
    pca_x = _subsample_rows(x, pca_max_cells)
    mean, scale, components = _fit_pca(
        pca_x, n_components=n_components, standardize=standardize
    )
    n_kept = int(components.size(1))
    out = x.new_zeros(num_classes, n_components)
    centered = class_means - mean
    if scale is not None:
        centered = centered / scale
    scores = centered @ components
    out[:, :n_kept] = scores
    out[~occupied] = 0
    return out


def _as_label_ids(labels: torch.Tensor) -> torch.Tensor:
    y = labels.detach()
    if y.dim() == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    if y.dim() != 1:
        raise ValueError(f"labels must be (N,) or (N, 1), got {tuple(labels.shape)}")
    return y.long()


def _class_means(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n_genes = int(features.size(1))
    sums = features.new_zeros(num_classes, n_genes)
    counts = features.new_zeros(num_classes, 1)
    valid = (labels >= 0) & (labels < num_classes)
    if not bool(valid.any()):
        occupied = torch.zeros(num_classes, dtype=torch.bool, device=features.device)
        return sums, occupied
    y = labels[valid]
    x = features[valid]
    sums.index_add_(0, y, x)
    ones = torch.ones(y.size(0), 1, device=features.device, dtype=features.dtype)
    counts.index_add_(0, y, ones)
    occupied = counts.squeeze(-1) > 0
    return sums / counts.clamp_min(1.0), occupied


def _subsample_rows(
    features: torch.Tensor, max_cells: Optional[int]
) -> torch.Tensor:
    if max_cells is None:
        return features
    cap = int(max_cells)
    if cap < 1 or features.size(0) <= cap:
        return features
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    idx = torch.randperm(features.size(0), generator=g)[:cap]
    return features[idx.to(features.device)]


def _fit_pca(
    features: torch.Tensor,
    n_components: int,
    *,
    standardize: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Return ``(mean, scale_or_None, components)`` with ``components`` ``(G, n)``.

    Components are leading eigenvectors of the (optionally z-scored) gene
    covariance, largest eigenvalue first. Uses a ``G×G`` eigendecomposition
    when ``N >= G``, otherwise an economy SVD of the centered matrix.
    """
    n_cells, n_genes = int(features.size(0)), int(features.size(1))
    mean = features.mean(dim=0)
    centered = features - mean
    scale: Optional[torch.Tensor] = None
    if standardize:
        scale = centered.std(dim=0, unbiased=False).clamp_min(1e-6)
        centered = centered / scale

    n_keep = min(n_components, n_genes, max(n_cells - 1, 1))
    if n_keep < 1:
        return mean, scale, features.new_zeros(n_genes, 0)

    work = centered if centered.dtype in (torch.float32, torch.float64) else centered.float()
    if n_cells >= n_genes:
        cov = (work.T @ work) / float(max(n_cells - 1, 1))
        _evals, evecs = torch.linalg.eigh(cov)
        components = evecs[:, -n_keep:].flip(1)
    else:
        _u, _s, vh = torch.linalg.svd(work, full_matrices=False)
        components = vh[:n_keep].T
    return mean, scale, components.to(dtype=features.dtype)


def _as_graph_inputs(
    features: torch.Tensor,
    domain_ids: torch.Tensor,
    node_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Normalize features / ids / mask to ``(B, N, ·)``."""
    if features.dim() == 2:
        features = features.unsqueeze(0)
        squeezed = True
    elif features.dim() == 3:
        squeezed = False
    else:
        raise ValueError(
            f"features must be (N, G) or (B, N, G), got {tuple(features.shape)}"
        )
    ids = domain_ids
    if ids.dim() == 3 and ids.size(-1) == 1:
        ids = ids.squeeze(-1)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError(
            f"domain_ids must be (N,) or (B, N), got {tuple(domain_ids.shape)}"
        )
    if tuple(ids.shape) != tuple(features.shape[:2]):
        raise ValueError(
            f"domain_ids shape {tuple(ids.shape)} does not match "
            f"features leading dims {tuple(features.shape[:2])}"
        )
    if node_mask is None:
        mask = torch.ones(
            features.shape[:2], device=features.device, dtype=features.dtype
        )
    else:
        mask = node_mask
        if mask.dim() == 3 and mask.size(-1) == 1:
            mask = mask.squeeze(-1)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != tuple(features.shape[:2]):
            raise ValueError(
                f"node_mask shape {tuple(mask.shape)} does not match "
                f"features leading dims {tuple(features.shape[:2])}"
            )
    return features, ids, mask, squeezed


def _scatter_mean(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Mean of ``src`` rows grouped by ``index`` → ``(dim_size, D)``."""
    n_dim = int(src.size(-1)) if src.dim() == 2 else 0
    if src.dim() != 2:
        raise ValueError(f"src must be (M, D), got {tuple(src.shape)}")
    out = src.new_zeros(dim_size, n_dim)
    if src.size(0) == 0:
        return out
    idx = index.long()
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), src)
    counts = src.new_zeros(dim_size, 1)
    counts.scatter_add_(
        0,
        idx.unsqueeze(-1),
        torch.ones(src.size(0), 1, device=src.device, dtype=src.dtype),
    )
    return out / counts.clamp_min(1.0)


def _scatter_attn_pool(
    src: torch.Tensor,
    scores: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Softmax-attention pool of ``src`` rows grouped by ``index``."""
    if src.dim() != 2:
        raise ValueError(f"src must be (M, D), got {tuple(src.shape)}")
    out = src.new_zeros(dim_size, src.size(-1))
    if src.size(0) == 0:
        return out
    idx = index.long()
    maxes = scores.new_full((dim_size,), float("-inf"))
    maxes.scatter_reduce_(0, idx, scores, reduce="amax", include_self=True)
    maxes = torch.nan_to_num(maxes, neginf=0.0)
    weights = (scores - maxes[idx]).exp()
    denom = scores.new_zeros(dim_size)
    denom.scatter_add_(0, idx, weights)
    weights = weights / denom[idx].clamp_min(1e-12)
    weighted = src * weights.unsqueeze(-1)
    out.scatter_add_(0, idx.unsqueeze(-1).expand_as(src), weighted)
    return out
