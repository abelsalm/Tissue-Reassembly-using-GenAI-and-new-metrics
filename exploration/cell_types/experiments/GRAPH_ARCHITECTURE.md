# Multi-scale CME and graph-restricted attention

Coordinate-free domain classifier. **Current best: `dom051_graph_ms` (val/acc 0.7706)**.
The model still never sees `x/y`. Spatial structure enters only as (1) precomputed Gaussian CME at multiple bandwidths and (2) a kNN graph built from CME (or transcriptome) *inside the 8192-cell bag*.

Ranking metric remains **val/acc**. Keep only if Δ ≥ ~0.5pp vs 012.

## Why these two changes

Domains are coarser than cell types and have sharp regional boundaries. Two gaps in 012:

1. **Scale.** Folder CME is a Gaussian-weighted type mix at a single bandwidth (σ=0.25, K=64). That is an *immediate-neighbor* microenvironment. Domain identity is a *regional* pattern. BANKSY (Singhal et al., Nat Genet 2024) makes the same distinction: low mixing weight λ for cell typing, high λ plus larger neighborhood kernels (`k_geom` 15–30, optionally two radii) for domain segmentation.
2. **Mixing.** 012 uses **linear attention** over the whole bag (`LinearAttentionTransformer`). Every cell can mix with every other cell in the 8192-chunk with no locality prior. STAGATE (Dong & Zhang, Nat Commun 2022) and GraphST (Long et al., Nat Commun 2023) instead message-pass on a *kNN graph*. They build that graph from physical coordinates; we cannot, so the graph is kNN in CME or gene-embedding space — the coordinate-free proxy for “who is transcriptomically / environmentally similar.”

Ideas 2 (iterative label refinement), 3 (supervised contrastive), and 5 (soft cluster then classify) are **not** in this wave.

## Idea 4 — multi-scale CME, per-scale residual add

### What failed before (`dom032_multiscale`, 0.7460)

σ = {0.12, 0.25, 0.5}, concatenated to 102-d, then crushed through the same 32-d CME MLP as single-scale 012. Three problems:

- σ=0.12 is barely larger than ABC median NN (~0.06); that scale is too fine to learn as an extra domain cue (and too close to type-level CME).
- Concat-then-bottleneck lets the fine scale dominate.
- K stayed 64 at every σ, so “coarse” CME was just a flatter weight on the *same* tiny neighborhood.

### What we do now

| σ | K | 3σ ball | role |
|---|---|---|---|
| 0.25 | 64 (existing) | 0.75 | keep 012 microenvironment |
| 1.0 | 256 | 3.0 | regional niche |
| 2.0 | 512 | 6.0 | domain-scale composition |

Each 34-d scale has its **own** SiLU MLP (34 → 32 → 32) and its **own** `Linear(32 → dx)`, **added** into gene tokens (same fusion as 012). No concat bottleneck. Type embed is unchanged.

Precompute: `experiments/precompute_abc_cme.py --sigma {1.0,2.0} --k {256,512}`. Coordinates used only offline. Files: `outputs/abc_soft_cmes/{train,validation,test}_sigma{1,2}.npz`.

Config: `experiments/configs/dom047_ms_coarse.json` (`aux_inputs.cme_sigmas = [0.25, 1.0, 2.0]`). `cme_n_scales` is inferred as `len(cme_sigmas)`.

Closest published analog: BANKSY’s product of own transcriptome × neighborhood transcriptome, with **multiple `k_geom`** and **high λ** for domains. We already add a neighborhood mix (CME); this wave only gives the domain head the coarser bandwidths BANKSY would use for segmentation rather than typing.

## Idea 1 — graph-restricted (kNN) attention

### What this is not

`gene_knn_k` (`dom021_knn16`, 0.7509) is a **head pooling** trick: cosine kNN in gene space, mean of neighbors’ mixed embeddings, concat onto the classifier. Attention itself stayed full-bag linear. This wave **replaces the attention kernel**.

### Graph

Inside each bag, build cosine kNN (`k` includes self) from **detached** features:

| `graph_on` | feature | meaning |
|---|---|---|
| `cme` | `data.cme_features` (concatenated scales if multi-scale) | cells with similar microenvironment / regional composition |
| `gene` | gene-MLP tokens **before** type/CME add | cells with similar transcriptome |

Indices are discrete (no grad through the graph). The same graph is reused across the 4 transformer layers — STAGATE-style *fixed neighbor set, learned attention weights*, not a dynamic graph that rewires every layer.

Pads never appear as neighbors. If a bag has fewer than `k` real cells, extra slots are masked with `−inf` before softmax.

### Attention

`GraphKNNAttention`: per head, scaled-dot attention of query *i* over its `k` neighbors only. Complexity of the mix is `O(N k d)` with `N=8192`, `k∈{16,32}`. Graph construction still uses a dense cosine matrix `O(N²)` inside the bag (~1 GB at batch 4, float32); that is acceptable on H200 and matches how 021 built its pool. We do **not** put extra tokens (encoder queries / aux tokens) in this mode — the sequence must stay cell-only.

This is closer to a **kNN-Transformer** (local scaled-dot) than classic GAT (concat attention logits). The locality prior is the same idea as STAGATE’s spatial neighbor network and GraphST’s GCN on a spatial kNN, with the spatial edges replaced by CME/transcriptomic edges.

### Configs

| ID | attention | graph | CME |
|---|---|---|---|
| **012** (control) | linear | — | σ=0.25 add |
| `dom047_ms_coarse` | linear | — | σ={0.25,1.0,2.0} per-scale add |
| `dom048_graph_cme16` | knn | CME, k=16 | σ=0.25 |
| `dom049_graph_cme32` | knn | CME, k=32 | σ=0.25 |
| `dom050_graph_gene32` | knn | gene, k=32 | σ=0.25 |
| `dom051_graph_ms` | knn | CME, k=32 | σ={0.25,1.0,2.0} per-scale add |

k=16–32 matches BANKSY’s usual `k_geom` range. The graph is bag-local; slice-scale structure is already baked into the CME *features* (precomputed on the full section).

Code: `knn_indices_from_features`, `GraphKNNAttention` in `ct_transformer.py`. Flags: `attn_kind=knn`, `graph_k`, `graph_on`.

## State of the art (how we differ)

| Method | Graph | Neighborhood features | Domain vs type |
|---|---|---|---|
| **BANKSY** (2024) | spatial kNN / rNN | weighted mean neighborhood transcriptome (+ optional AGF gradients); mix with λ | high λ → domains |
| **STAGATE** (2022) | spatial kNN, optional prune by expression | GAT autoencoder on that graph | unsupervised domains |
| **GraphST** (2023) | spatial kNN | GCN + contrastive (neighbors vs non-neighbors) | unsupervised domains |
| **SpaGCN** | spatial + histology | GCN | unsupervised domains |
| **012 + this wave** | **CME or gene kNN in the bag** (no x/y in the net) | Gaussian type-mix CME at one or more σ, **added** to gene tokens; **supervised** 47-way CE | supervised ABC `spatial_module_l1` |

We cannot copy STAGATE/GraphST’s spatial graph into the forward pass. CME kNN is the legitimate substitute: two cells that share a coarse type-mix are the ones that should exchange domain messages.

## Implementation notes

- Backbone unchanged: 4L, dx=128, 8 heads, ff=256, type embed 32, add fusion, SiLU, CE ls=0.05, AdamW 2e-4.
- `cme_n_scales=1` reproduces 012’s single CME MLP (ModuleList of length 1).
- Linear-attention 012/047 ≈ 1.35M params. kNN attention replaces `LinearAttentionTransformer` with QKV over `k` neighbors (~0.76M params) — different inductive bias, not a width/depth shrink.
- Dropout on type/CME **tokens** does not drop the CME graph: topology stays, injected features may zero.
- Next lever if boundaries stay blurry after this wave: idea 2 (2–3 rounds of domain-distribution message passing on the same graph).

## Results

**New best: `dom051_graph_ms` = 0.7706** (multi-scale CME + CME kNN k=32). Idea 4 and idea 1 **stack** (+0.70pp on top of 047). CME-space graph beats gene-space graph. k=32 slightly beats k=16.

| ID | val/acc | Δ vs 012 | Δ vs 047 | decision |
|---|---|---|---|---|
| **dom047_ms_coarse** | **0.7636** | **+1.10** | 0 | KEEP (idea 4) |
| **dom048_graph_cme16** | **0.7570** | **+0.44** | −0.66 | graph helps vs 012; under keep vs 047 |
| **dom049_graph_cme32** | **0.7584** | **+0.58** | −0.52 | best graph-only; loses to 047 |
| **dom050_graph_gene32** | **0.7532** | **+0.06** | −1.04 | drop; use CME graph not gene |
| **dom051_graph_ms** | **0.7706** | **+1.80** | **+0.70** | **KEEP; new canonical** |
