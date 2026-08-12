# exp010_gene_sim_aux

## Hypothesis

An auxiliary CME target built only from within-bag gene cosine similarity (no spatial coordinates) can push the CME head toward transcriptomic neighborhood structure, complementing the spatial σ=96 soft_cme target during training while keeping canonical eval unchanged.

## Changes vs exp006 (deeper wide CME)

- **Loss:** `gene_sim_cme_weight=0.5`, `gene_sim_temperature=0.5`, `gene_sim_source="label"` — per-cell target is a temperature-softmax weighted average of neighbors' one-hot types from raw gene features
- **Unchanged:** `cme_weight=1.5`, model (4-layer, wide CME, feature_cross, subgraph_summary), cosine_warmup scheduler
- **Eval:** unchanged — `build_canonical_eval_loss` uses CE(cls) + softCE(cme, σ=96) only; gene-sim aux is training-only

## Expected outcome

Compare `val/eval_loss` and `val/cme_loss` against exp006. Watch whether the gene-sim aux lowers eval CME without hurting cls accuracy. Training will include an extra `cme_gene_sim` term (not logged to wandb by default yet). N=1 subgraphs contribute zero gene-sim loss.
