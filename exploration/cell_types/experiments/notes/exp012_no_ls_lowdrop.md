# exp012_no_ls_lowdrop

## Hypothesis

Label smoothing (`ls=0.05`) inflates the CE floor during training, making the cls term harder to minimize; removing LS plus slightly lower dropout may reduce train cls loss and improve generalization on the fixed eval metric.

## Changes vs exp006 (deeper wide CME)

- **Loss:** `label_smoothing=0.0` (train only)
- **Model dropout:** `dropout_cls=0.25`, `dropout_layer=0.15`, `dropout_cme=0.05`, `dropout_fusion=0.05` (down from 0.4 / 0.25 / 0.1 / 0.1)
- **Unchanged:** `cme_weight=1.5`, architecture dims, feature_cross, subgraph_summary, cosine_warmup, early stopping
- **Eval:** canonical — always CE(cls, **ls=0.05**) + softCE(cme, σ=96), hardcoded in `build_canonical_eval_loss` regardless of training `label_smoothing`

## Expected outcome

Compare `val/eval_loss`, `val/cls_loss`, and per-class accuracy against exp006. Train cls loss should sit lower (no LS floor); eval cls is still measured with ls=0.05 so runs remain comparable. Watch for overfitting with reduced dropout.
