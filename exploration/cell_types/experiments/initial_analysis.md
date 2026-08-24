# Domain classifier — initial analysis

**Canonical metric:** `val/acc` (micro, all real cells). Loss may change; we always rank on this.

**Data.** MERFISH ABC, mouse-held-out: train ABC1 (2.85M cells), val/test ABC2 (1.23M). 1121 genes. Labels = `spatial_module_l1_complete`; train vocab = 46 `shared_*` + 1 `other` (C=47). Val `other` = 13.9% (NaNs + ABC2-only modules). Chunk size 8192; graphs rechunked every epoch **within section** (not spatial kNN). **Coordinates are never model inputs.**

**Existing runs (same script, `ct_config_domain.json` lineage).**

| run | params | best val/acc | train/acc @ best | notes |
|---|---|---|---|---|
| `20260819_155206` | dx=64, 4L, no bag summary, drop 0.3/0.25, wd=1e-3 | 0.6589 @ ep38 | 0.773 | still climbing; interrupted |
| **`20260819_161502`** | dx=128, 4L, concat skip + mean bag summary, drop 0.1, wd=1e-4, lr=2e-4 | **0.6726 @ ep112** | 0.820 | ES patience 16 val-checks (min_delta=0.001) |

Baseline to beat: **0.6726**. A change is kept only if Δ ≥ **~0.5pp** (→ ≥ 0.6776).

**Dynamics.** Val plateaus ~ep32–48 (~0.67) while train keeps rising to 0.82. Gap ≈ **15pp** = overfitting + ABC1→ABC2 shift. Val loss bottoms ~1.44 then drifts up. Macro acc ≈ 0.47 vs micro 0.67: 15 rare classes (n<~200 on val) are ~0%; abundant classes hit 0.82–0.92.

**Model.** Gene MLP → linear-attention set transformer → concat(`E_raw`,`E_mixed`) + bag mean → CE (ls=0.05). ~1.34M params. Transformer FFN is **ReLU** even though gene/cls heads are SiLU (`layer_activation` not forwarded). Attention **does not mask pads** (`attention_use_node_mask=false`). No LR schedule, no EMA.

**Allowed extra inputs (not targets).** CSV `cell_class` = 34 ABC types. Folder CME = Gaussian neighborhood type-mix (precomputed offline; model still sees a vector, not x/y). ABC coords are ~O(1), median NN ≈ 0.06 (not cortex microns); use σ≈0.25 not 96.

**Priorities.**
1. Generalization (gap 15pp): pad-mask, EMA, dropout/wd/lr, cosine.
2. Inductive bias: type embedding, CME vector, FiLM / head fusion.
3. Capacity only if (1–2) move ≥ 0.5pp.

**Ops.** ~44 min/run on H200 including data load. GPUs **6 and 7 only**. Dedicated configs under `experiments/configs/`. Compact `history.json` (no per-class dumps). Max 100 runs.
