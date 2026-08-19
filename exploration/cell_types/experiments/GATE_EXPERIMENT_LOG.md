# Gate research — experiment log (compact)

## Objective
Canonical `val/cme_loss` = SoftCE(σ=96). Stop if **< 2.1** or after wide gating search.
Eval SoftCE uses **final gated mix** when gate on. GT SoftCE floor ≈ **1.970**.

## Best
| id | CME | eval | recipe |
|---|---|---|---|
| **gate020_bag2_joint** | **2.2039** | 2.6124 | gated_joint SoftCE + bag_cme_weight=2, no presence BCE |

Δ vs ungated ~2.212 ≈ **0.37%** (< 0.5% keep-threshold). Far from 2.1.

## Dropped directions (Δ < 0.5% or worse)
| family | examples | CME | why drop |
|---|---|---|---|
| SoftCE source both | gate003 | 2.25 | fights π |
| presence BCE modes | 001/002/005 | 2.206 | ≈ wall |
| JS / reverse KL | 006/011 | 2.26–2.72 | geometry mismatch |
| support_eps / tgt_temp / classbal | 007–009 | 2.4–2.56 | hurts eval SoftCE |
| train SoftCE @σ≠96 | 014 | 2.22 | hurts σ=96 eval |
| hard top-k presence | 017 | 2.23 | too hard support |
| spearman / gene_sim / entmatch | 019/022/023 | 2.21–2.24 | no win |
| cme_weight=8 | 012 | — | wrecks CLS |
| presence bias init | 013 | 2.205 | noop |

## Code added during sweep
- `cme_softce_source=gated_joint` (grads through π⊙q)
- canonical eval SoftCE on gated mix
- `cme_target_support_eps`, `presence_sparsity_weight`, `cme_class_balance`
- `cme_train_sigma`, `cme_divergence=reverse_kl`
- `presence_bias_init`, `presence_topk`
- compact `summarize_run.py` + `GATE_SWEEP.jsonl`

## Conjecture (major)
Presence gating does **not** unlock the SoftCE gap to 2.1. SoftCE@σ=96 under this transformer capacity saturates ~**2.204** whether SoftCE is on ungated logits or gated mixes; KD historically only reached **2.197**. Closing ~0.1 nats more needs capacity / teachers / a different CME inductive bias — not presence-loss reweighting.
