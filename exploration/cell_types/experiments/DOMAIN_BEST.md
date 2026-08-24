# Best domain experiment

| Field | Value |
|---|---|
| ID | **dom051_graph_ms** |
| **val/acc** | **0.7706** |
| val/loss | 1.3307 |
| train/acc @ best | 0.9242 |
| train/loss @ best | 0.5871 |
| best epoch | 136 (ES stop ep196) |
| vs gene-only baseline | **+9.80pp** (0.6726 → 0.7706) |
| vs dom012 (type+σ=0.25 CME) | **+1.80pp** |
| vs dom047 (multi-scale CME only) | **+0.70pp** |
| Config | `experiments/configs/dom051_graph_ms.json` |
| Checkpoint | `outputs/dom051_graph_ms/best.pt` |
| Logs | `logs/dom051_graph_ms/20260821_032613/` |
| Canonical config | `ct_config_domain.json` (synced to this recipe) |

## Recipe
- 4-layer set transformer, dx=128, 8 heads, concat global skip, bag mean summary, SiLU gene/cls, CE ls=0.05, AdamW lr=2e-4, wd=1e-4.
- **Cell-type embedding** (34 ABC types, dim 32, 10% input dropout) **added** into gene tokens.
- **Multi-scale Gaussian kNN CME**, each scale with its own MLP and add-into-dx residual:
  - σ=0.25, K=64
  - σ=1.0, K=256
  - σ=2.0, K=512
- **Graph-restricted attention** (`attn_kind=knn`): cosine kNN in **CME space** (concatenated scales), k=32 including self, scaled-dot over neighbors only. No x/y in the forward pass.

Coordinates are never model inputs. CME is precomputed offline. Design: `GRAPH_ARCHITECTURE.md`.

## Ablations (wave 6)
| ID | change vs 012 | val/acc | vs 012 |
|---|---|---|---|
| 047 | multi-scale CME, linear attn | 0.7636 | +1.10 **keep** |
| 048 | CME graph k=16 | 0.7570 | +0.44 |
| 049 | CME graph k=32 | 0.7584 | +0.58 |
| 050 | gene graph k=32 | 0.7532 | +0.06 |
| **051** | **047 + CME graph k=32** | **0.7706** | **+1.80 keep** |

Multi-scale CME and CME-graph attention **stack**. Gene-space graph is weaker than CME-space graph.
