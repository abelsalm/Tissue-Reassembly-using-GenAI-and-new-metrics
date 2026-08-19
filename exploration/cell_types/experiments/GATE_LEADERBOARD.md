# Gate sweep leaderboard (final)

GT SoftCE floor ≈ **1.970** | Ungated ~**2.212** | **Best gated 2.2039** | Goal **< 2.1** ❌

| rank | id | CME | eval | notes |
|---|---|---|---|---|
| 1 | **gate020_bag2_joint** | **2.2039** | 2.6124 | gated_joint + bag=2 |
| 2 | gate010 sparse0.5 | 2.2044 | 2.6122 | sparsity prior |
| 3 | gate004 joint nopres | 2.2048 | 2.6125 | SoftCE through π⊙q |
| 4 | gate001/002 | 2.206 | 2.613 | detach SoftCE + BCE |
| — | many fails | ≥2.21 | | see GATE_EXPERIMENT_LOG.md |

**Verdict:** presence-gate SoftCE saturates ~2.204 under this arch. 0.5% wins not found; 2.1 not reached. See `GATE_EXPERIMENT_LOG.md` for conjecture.
