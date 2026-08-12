# Initial Analysis — Cell-Type Transformer + Soft CME

**Date:** 2026-08-07  
**Canonical evaluation metric (fixed across all experiments):**

\[
\mathcal{L}_{\mathrm{eval}} = \mathrm{CE}(\hat y_{\mathrm{cls}}, y) + \mathrm{softCE}(\hat y_{\mathrm{cme}}, p_{\sigma=96})
\]

- CE = masked hard cross-entropy (label_smoothing=0.05, matching current baseline reporting)
- softCE targets always from precomputed soft CME with **σ = 96**
- Stage-1 aux losses / train reweighting may change; **eval never does**

Reported as `val/eval_loss = val/cls_loss + val/cme_loss`.

---

## Pipeline summary

| Component | Current state |
|-----------|---------------|
| Model | `CellTypeTransformer`: gene MLP → 2× linear-attention layers → feature-cross towers (type/env) → fusion → stage-2 heads |
| Inputs used | **Transcriptomics only** (`node_features`). Positions are available in the batch but **must not** be used — spatial info is intentionally excluded from priors. |
| Targets | Hard cell class + soft microenvironment distribution (σ=96) |
| Loss (train) | `combined`: cls + cme + 0.5×(stage1 cls + stage1 cme) |
| Optimizer | AdamW, lr=2e-4, wd=1e-3, grad_clip=1.0 |
| Scheduler | **None** |
| Data | MERFISH small cortex; graphs of ≤2048 cells; rechunk every epoch |
| Early stop | `val/loss` (includes stage-1 weights), patience=16 val checks |

~212k parameters. Batch size 8. ~15–20s/epoch on H200 → full run ~1–2h with early stopping.

---

## Baseline performance (existing `outputs/ct_transformer`)

| Metric | Best | Epoch |
|--------|------|-------|
| **val/eval_loss (cls+cme)** | **2.6717** | 210 |
| val/cls_loss | 0.4449 | 210 |
| val/cme_loss | 2.2268 | 210 |
| val/acc | 0.9756 | 174 |
| train/acc (late) | ~0.985 | — |

### Information-theoretic CME floor

Validation soft-CME (σ=96): mean entropy ≈ **1.970** (uniform would be 3.135).  
Best cme_loss 2.227 → **gap ≈ 0.257 nats** to the Bayes floor.  
CLS is nearly saturated (~97.6% acc); **almost all remaining eval headroom is CME**.

---

## Bottlenecks / observations

1. **Transcriptomics-only constraint (hard).** Soft CME targets are spatial, but the model may only use gene expression of cells in the bag. It must recover microenvironment structure from co-expression / set structure — that is the research challenge. **Do not add position inputs.**
2. **CME dominates eval.** cls≈0.45 vs cme≈2.23; optimizing only accuracy leaves CME under-prioritized.
3. **Tiny capacity.** dx=64, gene MLP hidden=32, 2 layers, ~212k params — may underfit CME structure.
4. **Constant LR.** Late training plateaus with early stopping on flat `val/loss`; cosine/warmup may help finalize CME.
5. **Feature-cross already present** with stage-1 probes — levers are capacity, set-pooling, attention depth, and loss design (not spatial features).
6. **Train/val gap on CME** is small (train cme≈2.19 vs val≈2.23) → not classic overfitting; more likely under-capacity / weak set-context for CME.

---

## Hypotheses to test (priority order)

| ID | Hypothesis | Change |
|----|------------|--------|
| H1 | CME needs more capacity | Wider env tower / cme head / dx |
| H2 | Eval-aware training helps | Raise `cme_weight`; early-stop on `val/eval_loss` |
| H3 | Schedule improves late CME | Cosine decay + warmup |
| H4 | Bag-level context helps CME | `subgraph_summary=true` |
| H5 | Deeper mixing helps neighborhood inference from genes | n_layers 3–4 |
| H6 | Train loss ≠ eval loss can still help | JS / reverse-KL / temperature on CME while eval stays softCE |
| H7 | Stronger gene encoder helps both heads | larger gene MLP / dx |

---

## Experiment protocol

- Configs live in `experiments/configs/expXXX_*.json`
- Unique `run_name` → unique checkpoint + log dirs
- GPUs: **only 4,5,6,7** (never 0–3)
- Primary decision metric: **best `val/eval_loss`**
- Secondary: val/cls, val/cme, val/acc
- Track results in `experiments/EXPERIMENT_LOG.md` and keep `experiments/BEST_EXPERIMENT.md` current
