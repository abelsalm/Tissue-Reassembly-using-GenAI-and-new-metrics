# exp005_js_wide

## Hypothesis

Training CME with Jensen–Shannon divergence (symmetric, bounded) instead of soft cross-entropy may improve CME calibration and reduce overconfident predictions, while canonical eval still reports soft CE for comparability with prior runs.

## Changes vs exp002 (wide + subgraph_summary)

- **Loss:** `cme_divergence="js"` for training CME terms (main + stage1); `cme_weight=2.0` (up from 1.5)
- **Eval:** unchanged — `build_canonical_eval_loss` always uses soft CE
- **Model:** same wide CME head + `subgraph_summary=true` as exp002
- **Unchanged:** cosine_warmup scheduler (`t_max=250`), early stopping on `val/eval_loss`

## Expected outcome

Compare `val/cme_loss` (soft CE metric) against exp002. Training `train/cme` will reflect JS and is not directly comparable to exp002's training curve. Watch whether JS training yields lower eval CME without hurting `val/cls_loss`.
