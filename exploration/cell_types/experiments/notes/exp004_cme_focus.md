# exp004_cme_focus

## Hypothesis

Upweighting CME in the training objective (while validation still uses unweighted cls+cme) closes the CME performance gap without collapsing cell-type accuracy.

## Changes vs exp001 (baseline)

- **Loss:** `cme_weight=2.5`, `stage1_cme_weight=1.0`, `stage1_cls_weight=0.25` (cls_weight and label smoothing unchanged)
- **Model:** `subgraph_summary=true` (baseline architecture otherwise)
- **Unchanged:** cosine_warmup scheduler (`t_max=250`), early stopping on `val/eval_loss`

## Expected outcome

Meaningful drop in `val/cme_loss` compared to exp001, with at most a modest increase in `val/cls_loss`. `val/eval_loss` may improve if CME was the dominant error term; watch per-class accuracy for rare types.
