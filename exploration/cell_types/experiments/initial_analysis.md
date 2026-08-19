# Gating research — initial analysis

## Baseline
- Ungated best CME (today recipe / aux): ~**2.212** (run `20260812_122943`)
- Prior gated tries (`presence_weight=0.5`, default soft_cme target, SoftCE on **comp**): CME ~**2.206** — Δ≈0.3% (below 0.5% keep-threshold)
- Historical KD+calib floor: CME **2.1966** (exp222) — not the gating stack
- **Stop goal**: canonical `val/cme_loss` **< 2.1**

## Gating mechanics
Final CME: `p = normalize(π ⊙ q)` with `π=σ(presence)`, `q=softmax(comp)`.
Train knobs (loss / soft_cme / `model.hidden_mlp_dims.presence` only):
- `presence_weight`, `presence_target` ∈ {soft_cme, soft_cme_thresh, soft_cme_power, mass}
- `presence_pos_weight`, `presence_saturation`, `presence_target_power/eps`
- `cme_softce_source` ∈ {comp, gated, both} + `cme_gated_weight`
- optional `cme_divergence=js`, presence MLP width

## Eval fix
Canonical eval SoftCE now scores the **final gated mix** when `cme_presence_gate=true` (not composition alone). Prior gated numbers SoftCE'd `q` and were not comparable to ungated final SoftCE.

## Conjecture (major, not 0.001 tweaks)
SoftCE must supervise **p** (gated), while presence BCE uses a **sparse** target (`soft_cme_thresh` / mass). SoftCE-on-comp + soft simplex presence barely moved CME; factorization only helps if π and SoftCE roles are complementary.

## Sweep policy
- ~4 widely spaced values per axis; drop a direction if Δ < **0.5%** CME
- Parallel GPUs **4/5/6/7** only; unique `run_name` per exp
- Compact summaries → `experiments/GATE_SWEEP.jsonl`
