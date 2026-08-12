# exp003_deeper_silu

## Hypothesis

Deeper attention mixing (4 transformer layers) plus SiLU in the transformer FFN improves CME quality inferred from transcriptomic co-occurrence, without sacrificing cell-type classification.

## Changes vs exp001 (baseline)

- **Depth & activations:** `n_layers=4`, `layer_activation=silu`, `gene_mlp_depth=3`
- **Width:** `dx=96`, `num_heads=8`, `dim_ffX=192`
- **MLP heads:** gene/cls/type_stage1=64; cme/env_stage1=192; fusion=128
- **Unchanged:** dropouts (cls 0.4, cme 0.1, fusion 0.1, layer 0.25), cosine_warmup scheduler (`t_max=250`), combined loss weights (cls/cme 1.0, stage1 0.5/0.5, label smoothing 0.05)

## Expected outcome

Lower `val/cme_loss` and better soft-CME alignment relative to exp001, with `val/cls_loss` and `val/acc` remaining competitive. If capacity helps, `val/eval_loss` should beat the baseline (~2.67).
