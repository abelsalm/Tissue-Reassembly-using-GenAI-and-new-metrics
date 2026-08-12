# exp011_cme_temp

## Hypothesis

Temperature scaling on CME logits during training (`logits / T` with `T > 1`) softens the predicted distribution before soft CE, encouraging calibration toward the σ=96 soft targets without changing the eval objective.

## Changes vs exp006 (deeper wide CME)

- **Loss:** `cme_temperature=1.5` — main and stage-1 CME terms use `softCE(cme_logits / 1.5, soft_cme)`; `cme_weight=1.5` unchanged
- **Unchanged:** model (4-layer, wide CME, feature_cross, subgraph_summary), `label_smoothing=0.05`, cosine_warmup scheduler, early stopping (`min_delta=0.0005`, `patience=20`)
- **Eval:** canonical — CE(cls, ls=0.05) + softCE(cme, raw logits, σ=96); training temperature does not apply at eval

## Expected outcome

Compare `val/eval_loss` and `val/cme_loss` against exp006. Softer training targets may reduce overconfident CME logits and improve eval CME without hurting cls accuracy. Training `cme` loss will differ from eval `cme_loss` when T≠1 (expected).
