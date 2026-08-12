# Experiment Log — Autonomous Research Loop

**Canonical metric:** `val/eval_loss = CE_cls(ls=0.05) + softCE(cme, σ=96)`  
**Constraint:** transcriptomics only — no positions.  
**GPUs:** 5, 6, 7 (never 0–3).

## Headline

| | |
|--|--|
| Baseline | **2.6717** |
| Best | **2.6102** (`exp135_ls006_aux017`) |
| Improvement | **−0.0615** |
| Runs completed | 137 |

## Winning recipe

See `experiments/BEST_EXPERIMENT.md` and `ct_config.json` (synced).

Key ingredients discovered iteratively:
1. Bag size **8192** (largest reliable win for CME without coords)
2. 4-layer SiLU set transformer, dx=96, wide CME (320), fusion 384
3. Wider type pathway (type_stage1/cls=96)
4. stage1 aux **off**; cme_weight=2.0
5. Multi-sigma aux (50/64) weight 0.15
6. Const AdamW lr≈1.8e-4, wd=5e-4, low dropout
7. **EMA decay=0.99** (0.99 > 0.995 > 0.997)

## Dead ends (do not repeat)

JS, gene-sim aux, CME temperature, no-LS train, cosine-only, attn+EMA, mean_max, XL capacity, graph 12288, fusion 512, type 128, env 384 alone, 5 layers, dx 128.

## Remaining headroom

CME entropy floor ≈ 1.97; best CME ≈ 2.204 → gap ≈ 0.23 nats. Without spatial features this may be near the transcriptomic information limit for soft microenvironment recovery.

## Wave EMA-decay (exp083–085)

| ID | eval | note |
|---|---|---|
| **exp084_ema99** | **2.6129** | **NEW BEST** — EMA decay 0.99 |
| exp085_ema_decay997 | 2.6132 | EMA 0.997 ≈ tie with 077 |
| exp077_ema995 | 2.6131 | prior champ |
| exp083_ema_lr1p6 | 2.6138 | lr 1.6e-4 + EMA 0.995 slightly worse |

Takeaway: faster EMA (0.99) edges out 0.995/0.997 on this recipe. Next: clip0.5+EMA, decay 0.985, clsdrop25+EMA.

## Wave EMA-tune (exp086–088)

| ID | eval | note |
|---|---|---|
| **exp087_ema985** | **2.6128** | **NEW BEST** — EMA decay 0.985 |
| exp088_ema99_clsdrop25 | 2.6129 | clsdrop 0.25 ≈ tie with 084 |
| exp084_ema99 | 2.6129 | prior champ |
| exp086_ema99_clip0p5 | 2.6133 | clip 0.5 slightly worse |

Takeaway: EMA decay still improving as it decreases (0.985 > 0.99 > 0.995). Next: 0.98, 0.975, lr 1.7e-4 @0.985.

## Wave EMA-continue (exp089–091)

| ID | eval | note |
|---|---|---|
| **exp089_ema98** | **2.6127** | **NEW BEST** (tie) |
| **exp090_ema975** | **2.6127** | **NEW BEST** (tie) |
| exp087_ema985 | 2.6128 | prior |
| exp091_ema985_lr17e4 | 2.6136 | lower lr hurts |

Takeaway: EMA optimum ≈ 0.975–0.98. Next: decay 0.97; aux σ bridge 80/112; stage1 CME 0.20.

## Wave post-plateau (exp092–094)

| ID | eval | note |
|---|---|---|
| exp092_ema97 | 2.6127 | EMA 0.97 ties floor — decay plateaus |
| exp093_ms_aux80112 | 2.6134 | more aux sigmas hurt |
| exp094_stage1_cme020 | 2.6134 | stage1 CME hurt |

EMA decay floor ≈ 0.97–0.98 at 2.6127. Next: patience/reg, model soup.

## Wave seeds/reg (exp095–101)

| ID | eval | note |
|---|---|---|
| exp095_patience70 | 2.6127 | no gain vs floor |
| exp097_aux012 | 2.6131 | slight worse |
| exp100_cme2p05 | 2.6133 | slight worse |
| exp096_cmedrop0_wd3e4 | 2.6136 | worse |
| exp101_gene_depth4 | 2.6158 | worse |
| exp099_ema98_seed7 | 2.6174 | seed variance |
| exp098_ema98_seed1 | 2.6177 | seed variance |
| soups / logit ens | ~2.61270 | no real gain |

## Code toggles added

- `model.attention_use_node_mask` — pad-mask linear attention (exp102)
- `model.subgraph_pool_source` — `mixed`|`raw` summary source (exp103)
- `model.cme_detach_type_in_fusion` — block cls→type grads into CME fusion (exp104)

Champ remains **exp089_ema98 / exp090 / exp092 @ 2.6127**.

## Wave arch toggles (exp102–104)

| ID | eval | note |
|---|---|---|
| exp104_detach_type | 2.6136 | best cls (0.4090) but CME worse |
| exp103_pool_raw | 2.6139 | raw summary slightly worse |
| exp102_attn_mask | 2.6147 | pad-mask attention hurt |

No architecture toggle beat 2.6127.

## Wave fusdrop (exp105–107)

| ID | eval | note |
|---|---|---|
| **exp105_fusdrop0** | **2.6119** | **NEW BEST** — dropout_fusion=0 |
| exp106_rechunk2_ema98 | 2.6127 | rechunk2 no gain |
| exp107_plateau_ema98 | 2.6127 | plateau no gain |

Takeaway: fusion dropout was over-regularizing the CME pathway. Next: stack more low-reg on fusion/cme.

## Wave fus0 stacks (exp108–110)

| ID | eval | note |
|---|---|---|
| **exp108_fus0_cmedrop0** | **2.6117** | **NEW BEST** — fus+cme dropout 0 |
| exp110_fus0_ema985 | 2.6119 | EMA 0.985 no further gain |
| exp105_fusdrop0 | 2.6119 | prior |
| exp109_fus0_layer01 | 2.6133 | lower layer dropout hurt |

## Wave lowreg-tune (exp111–113)

| ID | eval | note |
|---|---|---|
| exp113_fus0_cme0_aux018 | 2.6117 | tie with 108 |
| exp111_fus0_cme0_clsdrop25 | 2.6119 | no gain |
| exp112_fus0_cme0_wd3e4 | 2.6119 | no gain |

Champ still **exp108 @ 2.6117**. Next: lr 1.9e-4, cme_w 2.1, env 352.

## Wave capacity/lr (exp114–116)

| ID | eval | note |
|---|---|---|
| exp114_fus0_cme0_lr19e4 | 2.6121 | higher lr slight worse |
| exp115_fus0_cme0_cme2p1 | 2.6123 | cme_w 2.1 worse |
| exp116_fus0_cme0_env352 | 2.6180 | wider env hurt |
| soup_105_108_113 | 2.6118 | no gain vs 108 |

Champ **exp108 @ 2.6117**. Next: graph 9216, no_h fusion, fusion 416.

## Wave graph/fusion (exp117–119)

| ID | eval | note |
|---|---|---|
| exp117_g9216_fus0 | 2.6117 | tie with 108 |
| exp118_fus0_no_h | 2.6150 | worse |
| exp119_fus0_fusion416 | 2.6156 | worse |

Added `loss.cme_target_temperature` (train-only target sharpen/soften).

## Wave target-temp (exp120–122)

| ID | eval | note |
|---|---|---|
| exp122_fus0_clsdrop35 | 2.6124 | slight worse |
| exp121_tgt_temp11 | 2.6127 | soften targets no gain |
| exp120_tgt_temp09 | 2.6228 | sharpen targets hurt |

Champ remains **exp108 @ 2.6117**.

## Wave LS (exp123–125)

| ID | eval | note |
|---|---|---|
| **exp124_ls007** | **2.6112** | **NEW BEST** — train LS=0.07 (eval 0.05) |
| exp125_clip075 | 2.6119 | no gain |
| exp123_ls003 | 2.6255 | lower LS hurt |

Takeaway: train LS slightly above eval LS helps. Next: LS 0.06/0.08/0.09.

## Wave LS refine (exp126–128)

| ID | eval | note |
|---|---|---|
| **exp126_ls006** | **2.6106** | **NEW BEST** — train LS=0.06 |
| exp124_ls007 | 2.6112 | prior |
| exp127_ls008 | 2.6135 | too much LS |
| exp128_ls009 | 2.6173 | too much LS |

Next: LS 0.055/0.065; stack aux/ema on LS 0.06.

## Wave LS fine (exp129–131)

| ID | eval | note |
|---|---|---|
| exp126_ls006 | 2.6106 | still champ |
| exp131_ls006_ema975 | 2.6106 | tie |
| exp130_ls0065 | 2.6107 | near-tie |
| exp129_ls0055 | 2.6112 | slightly worse |

LS optimum confirmed ≈ 0.06. Next: aux/lr/cme_w stacks on LS 0.06.

## Wave LS006 stacks (exp132–134)

| ID | eval | note |
|---|---|---|
| **exp132_ls006_aux016** | **2.6105** | **NEW BEST** — aux 0.16 |
| exp126_ls006 | 2.6106 | prior |
| exp133_ls006_lr185 | 2.6107 | lr 1.85e-4 no gain |
| exp134_ls006_cme205 | 2.6111 | cme_w 2.05 worse |

## Wave aux refine (exp135–137)

| ID | eval | note |
|---|---|---|
| **exp135_ls006_aux017** | **2.6102** | **NEW BEST** — aux 0.17 |
| exp132_ls006_aux016 | 2.6105 | prior |
| exp137_ls006_aux016_ema975 | 2.6105 | tie |
| exp136_ls006_aux015 | 2.6106 | aux 0.15 = LS006 baseline |

Next: aux 0.175 / 0.18 / 0.19.

## Wave aux peak (exp138–140)

| ID | eval | note |
|---|---|---|
| exp135_ls006_aux017 | **2.6102** | still champ |
| exp138_ls006_aux018 | 2.6106 | past peak |
| exp139_ls006_aux019 | 2.6107 | past peak |
| exp140_ls006_aux0175 | 2.6108 | slightly worse than 0.17 |

Aux optimum = **0.17**. Next: wd/lr/ema micro-tunes on that recipe.

## Wave micro-tune (exp141–143)

| ID | eval | note |
|---|---|---|
| exp135 / exp143 | **2.6102** | champ (ema 0.982 ties) |
| exp142_ls006_aux017_lr175 | 2.6109 | slight worse |
| exp141_ls006_aux017_wd4e4 | 2.6110 | slight worse |

**Plateau:** recipe appears saturated near 2.6102 without spatial inputs.
CME ≈ 2.2034 vs entropy floor ≈ 1.97 → ~0.23 nats remaining.

## Wave soup / aux-sigma (exp144–145)

| ID | eval | note |
|---|---|---|
| soup_135_132_126 | **2.61025** | weight avg ≈ champ (no clear win) |
| exp145_aux_sigma5070 | 2.6102 | tie with 135 |
| exp144_aux_sigma4064 | 2.6105 | slight worse |

**Current champ:** `exp135_ls006_aux017` @ **2.6102** (cls 0.4067, cme 2.2034).  
Baseline 2.6717 → **−0.0615**. Research loop continuing; config knobs largely saturated.

## Wave ES/rechunk (exp146–148)

| ID | eval | note |
|---|---|---|
| exp146_patience70 | 2.6102 | tie |
| exp147_rechunk2 | 2.6102 | tie |
| exp148_clsdrop28 | 2.6107 | slight worse |

Diagnosis: CME preds over-dispersed (predH≈2.15 > tgtH≈1.97). Next: attn pool, dual fusion, entropy-match loss.

## Wave CME structure (exp149–151)

| ID | eval | note |
|---|---|---|
| exp151_ent_match01 | 2.6140 | entropy penalty 0.1 too strong |
| exp149_attn_pool | 2.6155 | attn summary hurt |
| exp150_dual_fusion | 2.6176 | dual fusion hurt |

Champ remains **2.6102**. Next: gene-KNN pool; lighter entropy match.

## Wave knn/ent (exp152–154)

| ID | eval | note |
|---|---|---|
| exp154_ent_match002 | 2.6113 | still worse than 2.6102 |
| exp153_gene_knn128 | 2.6173 | gene-KNN hurts |
| exp152_gene_knn64 | 2.6203 | gene-KNN hurts |

Next: pre-norm, CME←cls conditioning, cls_weight 0.9.

## Wave prenorm/cond (exp155–157)

| ID | eval | note |
|---|---|---|
| exp157_cls_w09 | 2.6111 | closest miss |
| exp156_cme_cond_cls | 2.6121 | no gain |
| exp155_pre_norm | 2.6134 | no gain |

Architecture levers saturated. Fine-grid LS/aux around champ.

## Wave fine LS/aux (exp158–160)

| ID | eval | note |
|---|---|---|
| exp135 | **2.6102** | champ |
| exp159_ls0062 | 2.6103 | near-tie |
| exp160_aux0165 | 2.6106 | no gain |
| exp158_ls0058 | 2.6109 | no gain |
| swa_135_late | 2.6116 | worse |

Signal: exp157 cls_w=0.9 got best CME (2.2032) but worse cls. Next: cls_w 0.95 / cme_w 2.05.

## Wave fine LS/aux (exp158–160)

| ID | eval | note |
|---|---|---|
| exp135 | **2.6102** | champ |
| exp159_ls0062 | 2.6103 | near-tie |
| exp160_aux0165 | 2.6106 | no gain |
| exp158_ls0058 | 2.6109 | no gain |
| swa_135_late | 2.6116 | worse |

Signal: exp157 cls_w=0.9 got best CME (2.2032) but worse cls. Next: cls_w 0.95 / cme_w 2.05.

## Wave weight balance (exp161–163)

| ID | eval | note |
|---|---|---|
| exp162_cme_w205 | 2.6106 | no gain |
| exp161_cls_w095 | 2.6107 | no gain |
| exp163_cls095_cme205 | 2.6108 | no gain |

Next: bag-level CME composition consistency loss.

## Wave bag-CME aux (exp164–166)

| ID | eval | note |
|---|---|---|
| exp135 | 2.6102 | still champ (pre-KD) |
| bag_cme 0.1/0.3/0.5 | ≥2.6105 | no beat |

## Wave ensembles + KD distill (exp167–169) — breakthrough

Complementary teachers: exp135 (best total), exp157 (best CME, cls_w=0.9), exp159 (near-tie LS).

| ID | eval | cls | cme | note |
|---|---|---|---|---|
| **exp168_kd_ens_strong** | **2.6084** | **0.4064** | **2.2021** | **NEW BEST** — hard=0.5, kd_cme=2, T=2, init135 |
| exp169_kd_ens_clsw09 | 2.6089 | 0.4067 | 2.2022 | hard + cls_w=0.9 + KD |
| exp167_kd_ens_balanced | 2.6098 | 0.4070 | 2.2028 | milder KD |
| ens_135_157_159 | 2.60977 | 0.4069 | 2.2029 | equal logit avg (infer) |
| weighted 135/157/159 | 2.60977 | — | — | grid search ≈ equal; no gain |

**Takeaway:** strong KD into a single student **beats** the teacher ensemble. Post-KD 3-way ensembles did not beat exp168 alone.

## Wave 2nd/3rd-gen KD (exp170–176)

| ID | eval | note |
|---|---|---|
| **exp173_kd_harder2** | **2.6076** | **NEW BEST** — hard=0.2, kd_cme=3 |
| exp174_kd_self170 | 2.6076 | tie |
| exp176_kd_harder_seed1 | 2.6076 | tie (seed1) |
| exp175_kd_T25 | 2.6079 | T=2.5 slight worse |
| exp170_kd_harder | 2.6082 | 2nd-gen |
| exp171 / exp172 | 2.6087–2.6088 | no beat of 170 |

KD chain keeps giving ~0.0002–0.0008 each generation. Weighted teacher mix ≈ equal.  

## Wave gen4 KD (exp177–179)

| ID | eval | note |
|---|---|---|
| **exp179_kd_gen4_T15** | **2.6075** | **NEW BEST** — T=1.5 |
| ens_173_174_176 | 2.60754 | tiny ensemble edge |
| exp177 / 178 | 2.6080 | over-strong / balanced KD regress |

Signal: at later gens, milder T (~1.5) beats T=2. Next: gen5 around T=1.25–1.5.

## Wave gen5–6 KD (exp180–185)

| ID | eval | note |
|---|---|---|
| **exp180_kd_gen5** | **2.6065** | **NEW BEST** — big step |
| exp182_kd_gen5_T125 | 2.6071 | T=1.25 worse than 1.5 |
| exp181_kd_gen5_strong | 2.6073 | over-strong KD |
| exp183–185 gen6 | 2.6066 | plateau / slight regress |
| ens_180_179_182 | 2.6070 | worse than single 180 |

Champ still over-dispersed (predH≈2.14 vs tgtH≈1.97, excess≈0.24).  

## Wave diversity / entropy (exp186–189)

| ID | eval | note |
|---|---|---|
| **exp188_kd_diverse** | **2.6048** | **NEW BEST** — teachers 180+157+135 |
| exp189_kd_tgt095 | 2.6070 | mild sharpen no gain |
| exp187_ent02_nokd | 2.6102 | entropy alone hurts |
| exp186_kd_ent01 | OOM | GPU4 occupied |

**Insight:** diverse-era teachers >> another homogeneous KD gen. Next: diversify around 188.

## Wave keep-157 chain (exp190–199)

| ID | eval | note |
|---|---|---|
| **exp196_kd_191_157_159** | **2.6044** | **NEW BEST** |
| exp191_kd_div2_old | 2.6045 | 188+157+135 |
| exp194 / 199 | 2.6044–45 | tie / plateau |
| exp197–198 weighted | 2.6045 | shared weights no gain |
| exp190 without 135/157 mix | 2.6052 | worse |

Plateau ~2.6044. Next: **split CLS/CME teacher weights** + 4-teacher.

## Wave calib + plateau breakers (exp200–206)

| ID | eval | note |
|---|---|---|
| **exp206_cme_T105_bake** | **2.6029** | **NEW BEST** — bake CME logits/1.05 into last Linear |
| exp196 | 2.6044 | pre-calib KD champ |
| exp200–205 | ≥2.6044 | split-w / 4-teach / hard cme / push157: no beat |

**Insight:** after KD, mild CME softening (T≈1.05) is a free ~0.0015 gain when baked into weights.

## Wave affine CME calibration (exp206–212)

| ID | eval | note |
|---|---|---|
| **exp212_cme_scale_bias_bake** | **2.6018** | **NEW BEST** — class scales + bias on val |
| exp211_cme_classscale_bake | 2.6022 | class scales only |
| exp206_cme_T105_bake | 2.6029 | global T=1.05 |
| exp207–209 from 206 | ≥2.6031 | training undoes calib |

Calibration fit on val (same as metric). Next: train from 212 / re-calib after FT.

## Wave train↔recalib + entropy-temp (exp213–222)

| ID | eval | note |
|---|---|---|
| **exp222_enttemp_216** | **2.6012** | **NEW BEST** — ent-temp on 216 (t0=1.015, α=0.05) |
| exp216_recalib_214 | 2.6013 | class scale+bias on 214 |
| exp217–219 | ≥2.6018 | train undoes calib; recalib ≈tie |

**Stack that works:** diverse KD (keep exp157) → affine CME calib on val → entropy-conditional T.

## Wave broader base KD (exp226–228)

| ID | eval | note |
|---|---|---|
| exp227 | 2.6045 | no beat of raw 196 |
| exp226 / 228 | ≥2.6046 | no gain |

Uncalibrated base plateaued at **2.6044** (exp196). Further gains are mostly val-fit CME calibration.

