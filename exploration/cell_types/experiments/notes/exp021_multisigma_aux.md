# exp021_multisigma_aux

## Hypothesis

Training the CME head against additional precomputed soft targets at smaller spatial scales (σ=50, σ=64) alongside the canonical σ=96 target can improve neighborhood-type calibration without changing the eval metric.

## Changes vs exp013 (no stage-1)

- **Soft CME:** primary `sigma=96`; aux targets `aux_sigmas=[50, 64]` attached as `batch.soft_cme_aux`
- **Loss:** `aux_cme_weight=0.3` — mean soft CE over aux sigmas, added to training total; primary `cme_weight=1.5` unchanged
- **Unchanged:** stage-1 weights 0, const LR, model architecture
- **Eval:** canonical — `build_canonical_eval_loss` uses CE(cls, ls=0.05) + softCE(cme, σ=96) only; aux sigmas are training-only

## Expected outcome

Compare `val/eval_loss` against exp013. Watch whether tighter-scale aux targets lower eval CME without hurting cls accuracy. Training logs include `cme_aux_loss` in epoch metrics.
