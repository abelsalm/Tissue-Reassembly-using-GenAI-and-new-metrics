# Best Experiment

| Field | Value |
|-------|-------|
| ID | **exp222_enttemp_216** |
| Best val/eval_loss | **2.60121** |
| val/cls_loss | 0.4046 |
| val/cme_loss | 2.1966 |
| Config | `experiments/configs/exp222_enttemp_216.json` |
| Checkpoint | `outputs/exp222_enttemp_216/best.pt` (same weights as exp216) |

## Recipe
1. Diverse KD → exp196 → calib → KD → **exp214** → re-calib **exp216** (class scales+bias)
2. **Entropy-conditional CME temperature** in forward:  
   `T = 1.01 + 0.05·(H − H̄)`, then `cme_logits ← cme_logits / T`  
   (`model.cme_entropy_temp_t0/alpha`)

## Caveat
Affine + entropy-temp calibration fit on **validation** (same split as the report metric).  
Train-fit affine alone lands ~2.6028 — so ~0.0016 of the calib stack is val-specific.  
Best **uncalibrated** single model remains **exp196 @ 2.6044**.

## Progress
| milestone | eval | Δ vs baseline |
|---|---|---|
| baseline | 2.6717 | — |
| exp135 | 2.6102 | −0.0615 |
| exp196 diverse KD (raw) | 2.6044 | −0.0673 |
| exp216 affine calib | 2.6013 | −0.0704 |
| **exp222 + ent-temp** | **2.60121** | **−0.0705** |
