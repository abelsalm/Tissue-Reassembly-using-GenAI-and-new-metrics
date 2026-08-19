# Best gating experiment

| Field | Value |
|---|---|
| ID | **gate020_bag2_joint** |
| val/cme_loss | **2.2039** |
| val/eval_loss | 2.6124 |
| val/cls_loss | 0.4085 |
| Config | `experiments/configs/gate020_bag2_joint.json` |
| Checkpoint | `outputs/gate020_bag2_joint/best.pt` |
| Logs | `logs/gate020_bag2_joint/20260812_193223/` |

## Recipe
- `cme_presence_gate=true`
- `cme_softce_source=gated_joint` (SoftCE through live π⊙q)
- `bag_cme_weight=2.0`
- `presence_weight=0` (no BCE)
- otherwise best_recipe defaults (cme_weight=2, aux=0.17, ls=0.06)

`ct_config.json` updated to this recipe (`run_name=ct_transformer_gate_best`).
