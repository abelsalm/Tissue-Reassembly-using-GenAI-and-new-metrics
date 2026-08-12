# exp007_attn_ema

## Hypothesis

Attention-pooled subgraph summaries (instead of masked mean) give a richer bag-level neighborhood signal for per-cell heads, and EMA-smoothed weights yield stabler validation metrics and better generalization on the combined cls + CME objective.

## Changes vs exp003 (deeper SiLU baseline)

- **Subgraph summary:** `subgraph_summary=true`, `subgraph_pool=attn` (learned masked softmax pool over transformer states, broadcast-concat)
- **EMA:** `train.ema.enabled=true`, `decay=0.999` — shadow weights updated each optimizer step; validation runs with EMA weights swapped in; checkpoints include `ema_state_dict`
- **Early stopping:** patience 20, `min_delta=0.0005` (slightly looser than exp003’s 0.001)
- **Unchanged from exp003:** 4-layer SiLU stack, gene_mlp_depth=3, widths, dropouts, cosine_warmup (`t_max=250`), combined loss (cls/cme 1.0, stage1 0.5/0.5, label smoothing 0.05)

## Expected outcome

Lower `val/eval_loss` vs exp003 if attention pooling captures non-uniform neighborhood structure better than mean pool. EMA should reduce val metric noise; if both help, `val/acc` and soft-CME alignment improve without hurting cls head.
