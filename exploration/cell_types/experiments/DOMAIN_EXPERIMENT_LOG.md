# Domain experiment log

Canonical metric: **val/acc**. Baseline **0.6726**. Keep if Δ ≥ ~0.5pp. GPUs 6/7.
Winner: `DOMAIN_BEST.md`. Narrative: `DOMAIN_SUMMARY.md`.

| ID | hypothesis | val/acc | decision |
|---|---|---|---|
| baseline `20260819_161502` | gene-only dx128 | 0.6726 | reference |
| prior-small | dx64 | 0.6589 | weaker |
| dom001_padmask | pad mask in attn | 0.6716 | drop |
| dom002_lr1e4 | lr=1e-4 | 0.6718 | drop |
| dom003_lr3e4 | lr=3e-4 | 0.6742 | drop LR |
| dom004_ema98 | EMA 0.98 | 0.6729 | drop |
| dom009_type_add | type embed add | 0.6784 | keep (+0.58) |
| **dom012_type_cme_add** | **type+CME add at input** | **0.7526** | **BEST +8.00** |
| dom013_film | FiLM | 0.7500 | drop |
| dom015_cme_only | CME, no type | 0.7460 | type needed |
| dom018_head | input+head | 0.7507 | drop |
| dom019_headonly | aux only at head | 0.7440 | drop |
| dom020_concat_in | concat fusion | 0.7456 | drop |
| dom021_knn16 | gene kNN=16 | 0.7509 | drop |
| dom022_cme_sigma05 | CME σ=0.5 | 0.7525 | drop σ |
| dom017_silu | SiLU FFN | 0.7503 | drop |
| dom023_prenorm | pre-LN | 0.7512 | drop |
| dom024_meanmax | bag mean+max | 0.7482 | drop |
| dom025_emb64 | type/CME embed 64 | 0.7526 | tie, drop |
| dom016_nodrop | no aux dropout | 0.7469 | drop (overfit) |

Around the winner, no change hit +0.5pp. Stopped.

## Wave 3 — scale up (no EMA / LR / dropout)

Go bigger on layers and embedding, then new cell representations.

| ID | hypothesis | val/acc | decision |
|---|---|---|---|
| dom026_dx256 | dx=256, ff=512, gene/cls=256, 4L | 0.7477 | drop (−0.49); width overfits |
| dom027_L6 | 6 layers, dx=128 | 0.7499 | drop (−0.27) |
| dom028_L6_dx256 | 6L + dx256 | skip | width already failed |
| dom029_L8 | 8 layers, dx=128 | 0.7510 | drop (−0.16); scale-up done |

Bigger capacity does not beat 4L dx128. Keep 012 size.

## Wave 4 — new cell / domain representations (012 backbone)

| ID | hypothesis | val/acc | decision |
|---|---|---|---|
| dom033_auxtok_fused | type+CME as extra tokens, *no* add | 0.6760 | drop; must keep add fusion |
| dom032_multiscale | CME σ={0.12,0.25,0.5} concat | 0.7460 | drop (−0.66) |
| dom030_queries8 | 8 learned domain prototypes at head | 0.7518 | drop (−0.08) |
| dom031_typehist | bag type histogram concat at head | 0.7472 | drop (−0.54) |
| dom035_encq16 | 16 encoder domain-query tokens | 0.7438 | drop (−0.88) |
| dom039_dualstream | parallel context transformer residual | 0.7444 | drop (−0.82) |
| dom038_gate | sigmoid gate on type+CME residual | 0.7512 | drop (−0.14) |
| dom040_genemlp3 | gene MLP depth 3, hidden 256, dx=128 | 0.7510 | drop (−0.16) |
| dom036_typeemb128 | type/CME embed 128 | 0.7434 | drop (−0.92); 64 was already a tie |

No new architecture hit +0.5pp vs 012. Best remains **dom012_type_cme_add (0.7526)**. Closest: queries 0.7518, gate 0.7512, L8 0.7510.

## Wave 5 — rare / zero-acc classes (012 backbone, 3 jobs/GPU)

Gene-only best epoch had 14/47 classes at 0 val acc (also 0 train). Ranking still **val/acc** micro; `best_per_class.json` now saved once at best.

| ID | hypothesis | GPU | val/acc | decision |
|---|---|---|---|---|
| dom041_balce | inverse-freq class weights | 6 | 0.3458 | drop (−40.7); destroys micro-acc |
| dom042_focal2 | focal γ=2 | 6 | 0.7483 | drop (−0.43); hurts micro-acc |
| dom043_ls0 | label_smoothing=0 | 6 | 0.7509 | drop (−0.17); keep ls=0.05 |
| dom044_invsqrt | inverse-sqrt class weights | 7 | 0.7416 | drop (−1.10); class weights hurt |
| dom045_focal1 | focal γ=1 | 7 | 0.7518 | drop (−0.08); closest, still <0.5pp |
| dom046_focal2_balce | focal γ=2 + inverse-freq | 7 | 0.7164 | drop (−3.62) |

Wave 5 done. **None kept.** Reweighting/focal trades majority-class acc for rare classes and loses on the ranking metric. Stick with unweighted CE, ls=0.05. `dom041` best-val still had 6/47 classes at 0 acc (including `shared_01`, n=96k).

## Wave 6 — multi-scale CME + graph attention (012 backbone)

Design: `GRAPH_ARCHITECTURE.md`. No tiny σ (drop 0.12). Per-scale add, not concat-bottleneck. Graph is CME/gene kNN, never x/y.

| ID | hypothesis | val/acc | decision |
|---|---|---|---|
| **dom047_ms_coarse** | **CME σ={0.25,1.0,2.0} per-scale add, K=64/256/512** | **0.7636** | **KEEP +1.10 vs 012** |
| **dom048_graph_cme16** | knn attn, graph_on=cme, k=16 | 0.7570 | +0.44 vs 012; under keep vs 047 |
| **dom049_graph_cme32** | knn attn, graph_on=cme, k=32 | 0.7584 | +0.58 vs 012; loses to 047 (−0.52) |
| **dom050_graph_gene32** | knn attn, graph_on=gene, k=32 | 0.7532 | +0.06 vs 012; CME graph > gene graph |
| **dom051_graph_ms** | **047 + CME kNN k=32** | **0.7706** | **KEEP +0.70 vs 047; new best** |

Wave 6 done. Multi-scale CME and CME-graph attention **stack**. Next lever if boundaries stay soft: idea 2 (iterative domain-distribution message passing on this graph).

