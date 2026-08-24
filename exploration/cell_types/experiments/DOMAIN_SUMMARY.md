# Domain classifier — full experiment summary

**Metric:** `val/acc` on ABC2 (held-out mouse). Labels: 46 `shared_*` spatial modules + `other`.
**Baseline:** gene-only transformer **0.6726** (`logs/ct_transformer_abc_domains/20260819_161502`).
**Best:** **dom051_graph_ms = 0.7706** (+9.80pp vs gene-only, +1.80pp vs 012, +0.70pp vs 047). Details in `DOMAIN_BEST.md`.
**Keep threshold:** ≥ 0.5pp. GPUs 6/7. No git commits.

## Leaderboard

| rank | ID | val/acc | Δ vs baseline | notes |
|---|---|---|---|---|
| 1 | **dom051_graph_ms** | **0.7706** | **+9.80** | multi-scale CME + CME kNN attn k=32 |
| 2 | **dom047_ms_coarse** | **0.7636** | **+9.10** | multi-scale CME only (linear attn) |
| 3 | dom049_graph_cme32 | 0.7584 | +8.58 | CME kNN k=32; +0.58 vs 012, loses to 047 |
| 4 | dom048_graph_cme16 | 0.7570 | +8.44 | CME kNN k=16 |
| 5 | **dom012_type_cme_add** | **0.7526** | **+8.00** | type + single-scale CME σ=0.25 |
| 2 | dom025_emb64 | 0.7526 | +8.00 | type/CME embed 64; tie, not kept |
| 3 | dom022_cme_sigma05 | 0.7525 | +7.99 | CME σ=0.5 |
| 4 | dom030_queries8 | 0.7518 | +7.92 | 8 domain prototypes at head |
| 5 | dom023_prenorm | 0.7512 | +7.86 | pre-LN |
| 5 | dom038_gate | 0.7512 | +7.86 | gated residual fusion |
| 7 | dom018_type_cme_head | 0.7507 | +7.81 | also concat aux at classifier head |
| 8 | baseline gene-only | 0.6726 | 0 | dx128, no type/CME |

## Wave 1 — hyperparameters / correctness (drop: all <0.5pp)

| ID | change | val/acc | decision |
|---|---|---|---|
| dom001_padmask | `attention_use_node_mask=true` | 0.6716 | drop |
| dom002_lr1e4 | lr=1e-4 | 0.6718 | drop |
| dom003_lr3e4 | lr=3e-4 | 0.6742 | drop LR search |
| dom004_ema98 | EMA 0.98 | 0.6729 | drop |

**Interpretation.** The 15pp train/val gap is mostly ABC1→ABC2 shift, not pad-leakage or LR/EMA. Capacity/regularization knobs on the gene-only model do not move micro-acc by 0.5pp.

## Wave 2 — architectures (type / microenvironment as *inputs*)

Never coordinates in the forward pass. CME = folder Gaussian neighborhood type-mix, precomputed (`outputs/abc_soft_cmes/`, σ=0.25, K=64).

| ID | architecture | val/acc | decision |
|---|---|---|---|
| dom009_type_add | type embed → add into gene tokens | 0.6784 | keep type (+0.58pp) |
| **dom012_type_cme_add** | type + CME add at input | **0.7526** | **best** |
| dom013_type_cme_film | FiLM(type,CME) on gene tokens | 0.7500 | drop vs add (−0.26) |
| dom015_cme_only | CME add, no type embed | 0.7460 | type still helps (−0.66 vs 012) |
| dom018_type_cme_head | add at input **and** concat at head | 0.7507 | drop (−0.19) |
| dom019_type_cme_headonly | transformer genes-only; aux at head | 0.7440 | drop (−0.86); mix aux *into* attention |

**Interpretation.** Spatial modules are largely neighborhood cell-type composition. CME alone jumps ~7.3pp; adding the cell’s own type adds another 0.66pp. Injecting that signal *before* attention beats using it only as a classifier feature. Further fusion variants (FiLM, extra head concat) are noise around the add-at-input recipe.

## Round 2 (GPUs 6+7, around the winner)
concat, gene-kNN, CME σ=0.5, SiLU FFN, pre-LN, mean+max bag, embed 64, no aux-dropout: all **≤ 0.7526**, none +0.5pp. Closest: emb64 **tie 0.7526**, prenorm 0.7512, kNN 0.7509.

## Wave 3 — scale up (no EMA / LR / dropout)

Bigger layers and embeddings, as requested. All **below** 012.

| ID | change | val/acc | vs 012 |
|---|---|---|---|
| dom026_dx256 | dx=256, ff=512 | 0.7477 | −0.49; overfits |
| dom027_L6 | 6 layers | 0.7499 | −0.27 |
| dom029_L8 | 8 layers | 0.7510 | −0.16 |
| dom036_typeemb128 | type/CME embed 128 | 0.7434 | −0.92 |
| dom040_genemlp3 | gene MLP 3×256 | 0.7510 | −0.16 |

**Interpretation.** The dataset is hard because of ABC1→ABC2 shift, not under-capacity. Width overfits; depth 6/8 is noise. Keep 4L dx128.

## Wave 4 — new cell / domain representations (012 backbone)

| ID | idea | val/acc | vs 012 |
|---|---|---|---|
| dom033_auxtok_fused | type+CME as extra tokens, drop add | 0.6760 | −7.66; add is required |
| dom032_multiscale | CME σ={0.12,0.25,0.5} | 0.7460 | −0.66 |
| dom030_queries8 | learned domain prototypes at head | 0.7518 | −0.08 |
| dom031_typehist | bag type histogram at head | 0.7472 | −0.54 |
| dom035_encq16 | encoder query tokens in the set | 0.7438 | −0.88 |
| dom039_dualstream | parallel context transformer | 0.7444 | −0.82 |
| dom038_gate | gated residual fusion | 0.7512 | −0.14 |

**Interpretation.** Unconditional **add of type + single-scale CME into gene tokens** is the representation. Replacing add with tokens collapses to gene-only. Extra prototypes, bag histograms, multi-scale CME, dual streams, and gates do not add 0.5pp. Closest miss: head queries (0.7518).

## Not worth further pursuit
HP (lr/EMA/dropout/wd); scaling width/depth; fusion/inject variants; σ, kNN, pre-LN, mean+max; replacing add with tokens; multi-scale CME; domain-query / type-hist heads.

## Code / data artifacts
- `ct_transformer.py`: type embedding, CME projection, add/concat/FiLM/gate, inject=input/head/both, optional domain queries, bag type hist, aux-as-tokens, encoder queries, dual-stream.
- `ct_train_domains.py`: compact history, aux lookup by `cell_ID` (no coords in the model), multi-scale CME concat.
- `experiments/precompute_abc_cme.py` + `outputs/abc_soft_cmes/`.
- Per-run configs: `experiments/configs/dom*.json`.
- Train with: `python exploration/cell_types/ct_train_domains.py --config exploration/cell_types/ct_config_domain.json` (still the 012 recipe).
