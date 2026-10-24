# Tier-F analysis — Avazu temporal calibration (2026-10-24)

Computed by `results/tierf/tierf_audit.py` from the Tier-F run outputs and saved predictions; nothing is hand-entered. Experiment B is the same-target-day identification control named as missing in the v15 threats ("The random and temporal test sets differ in dates, hours, base rate, and size; a same-target-day comparison would isolate the protocol effect"). Experiment A (the bidding-parameter sweep) is spot-checked in §6.

**Design, as executed** (`tierf_prep.py` / `tierf_same_day.py`, chain `run_tierf.sh`). For each origin (1, 2) and seed (42, 43, 44), the control arm draws uniformly at random **without replacement from the origin's own five-day training window** — the same pool the temporal arm stride-samples — with the row count **exactly equal** to the temporal arm's training size (origin 1: 3,338,217 of 20,029,297 window rows; origin 2: 3,948,439 of 19,742,194; draw rule `default_rng(seed).choice(window, n, replace=False)`, sorted). The vocabulary encoder is refit on the drawn rows only, mirroring the per-arm encoder protocol of every other arm. One model per (origin, seed, model) is trained with the shared pipeline and scored **raw** on the same four target days as the temporal arm; test buckets are row-identical to the temporal arm's (sizes asserted against the temporal manifests in prep; `n_test` asserted again at fit time). The temporal arm is never retrained; its metrics are loaded from its own runs' `metrics_temporal.csv` (variant `raw`). 18 fits, 72 paired cells. Deltas below are Δ = random-draw arm − temporal arm on the *same* target day; for AUC positive favours the draw, for ECE negative favours the draw.

## 1. Integrity and recomputation

- **Grid.** All 18 cell CSVs present, 4 rows each (gaps 0–3), 72 rows total, no NaN metrics. `n_train` is constant per origin and equals the size-match target in all 72 rows (3,338,217 / 3,948,439). `n_test` is identical across models and seeds for each (origin, gap) and equals the tierf arm manifests' split sizes, which in turn assert against the temporal manifests.
- **Size match in the logs.** All six prep logs record the window size and the drawn count (20,029,297 → 3,338,217; 19,742,194 → 3,948,439). 17 of 18 fit logs carry the line `train rows: N (temporal arm: N)` with both numbers equal to the target; the exception is `o1_s42_lr`, whose step log is the handwritten note of the manual smoke run that produced that cell before the chain existed — its `n_train` is corroborated by its cell CSV, its arm manifest, and its prep log.
- **Temporal columns.** The `temporal_auc / temporal_logloss / temporal_ece / n_test` columns embedded in the cell CSVs match their six source files (`results/o1_repro`, `results/seeds/seed{43,44}`, `results/origin2`, `results/origin2_seeds/seed{43,44}` — all `metrics_temporal.csv`, variant `raw`) with max abs difference **0.0** on every quantity.
- **Recomputation from saved predictions.** AUC (sklearn `roc_auc_score`), log loss, and ECE-15 (independent `np.bincount` implementation of the project convention: clip to [1e-7, 1−1e-7], bin = min(int(p·15), 14), count-weighted |frac_pos − mean_pred|) were recomputed for all 72 (fit, gap) pairs from `results/tierf/preds/o{origin}_s{seed}/tierf_{model}_gap{g}_raw.npy` against the arm's own test labels. **Max |recomputed − CSV| = 0.0 (AUC), 9.5e-17 (ECE), 5.6e-17 (log loss)** — exact reproduction, below even the float32-storage floor of the Tier-C check (1.5e-9), because these models emit float32 probabilities natively and the runs' metrics were computed on those same values. All 72 prediction files have byte size exactly 128 + 4·n_test, lengths equal to `n_test`, all values finite and inside [2.6e-05, 0.99642].
- **Restart forensics.** The chain was started 10 times (runtime restarts); the log ends with `TIERF CHAIN COMPLETE`, **no step ever finished with rc ≠ 0**, and exactly one finish is recorded per step (24/24: 6 preps + 18 fits). Five steps were *started* more than once: `o1_s42_dcnv2` ×2, `o2_s42_deepfm` ×3, `o2_s42_dcnv2` ×3, `prep_o2_s44` ×2, `o2_s44_dcnv2` ×4 (its first start was 2026-10-24T02:24:25Z; the surviving run started 02:52:55Z and finished rc=0 at 03:09:52Z). For each multi-start fit, the four on-disk prediction files carry strictly increasing mtimes in gap order terminating at the cell CSV's mtime (e.g. `o2_s44_dcnv2`: preds 03:05:51 → 03:09:49Z, CSV 03:09:49Z; its step log is the final run's and its printed per-gap metrics match the CSV). Combined with the exact recomputation above, the saved predictions and the cell CSVs are provably one single run's outputs for every cell: **no duplicated or concurrent run left any trace in the artifacts.**

## 2. Headline comparison (Δ = draw − temporal, same target days)

| scope | n | mean ΔAUC | AUC wins (draw/temp) | mean ΔECE | ECE wins (draw/temp) | mean Δlogloss |
|---|---|---|---|---|---|---|
| pooled | 72 | +0.00015 | 43 / 29 | +0.00053 | 35 / 37 | −0.00009 |
| origin 1 | 36 | −0.00015 | 22 / 14 | +0.00039 | 17 / 19 | +0.00004 |
| origin 2 | 36 | +0.00046 | 21 / 15 | +0.00066 | 18 / 18 | −0.00023 |
| seed 42 | 24 | −0.00019 | 13 / 11 | +0.00013 | 11 / 13 | −0.00006 |
| seed 43 | 24 | −0.00029 | 11 / 13 | +0.00120 | 10 / 14 | +0.00019 |
| seed 44 | 24 | +0.00095 | 19 / 5 | +0.00024 | 14 / 10 | −0.00040 |
| LR | 24 | +0.00021 | 15 / 9 | +0.00052 | 13 / 11 | −0.00003 |
| DeepFM | 24 | −0.00051 | 11 / 13 | +0.00072 | 10 / 14 | +0.00019 |
| DCN-v2 | 24 | +0.00076 | 17 / 7 | +0.00033 | 12 / 12 | −0.00044 |

At **gap 0** — the bucket RQ1 headlines — pooled over its 18 cells: mean ΔAUC **+0.00012** (draw wins 10/18), mean ΔECE **−0.00097** (draw wins 12/18). Per model at gap 0 (6 cells each): LR ΔAUC +0.00020, ΔECE −0.00065 (draw better on ECE in 5 of 6 origin×seed cells, per-cell ΔECE ∈ [−0.00116, +0.00004], ΔAUC ∈ [−0.00054, +0.00074]); DeepFM ΔAUC −0.00008, ΔECE −0.00129; DCN-v2 ΔAUC +0.00022, ΔECE −0.00098.

Contrast with RQ1's random **split** at origin 1 gap 0 (paper Table 1 / Tier-C §3b): ΔAUC +0.0135 / +0.0139 / +0.0141 (LR/DeepFM/DCN-v2, seed 42; +0.0135 to +0.0163 across seeds), and ECE reported at 55% / 55% / 74% of the temporal value (absolute ΔECE ≈ −0.0041 / −0.0031 / −0.0019). The same-day, exactly size-matched draw leaves a residual two orders of magnitude smaller on AUC (≈1% of the split contrast) and, for LR at origin 1 gap 0, an ECE delta of −0.0003/−0.0003/+0.0000 across seeds against the split's −0.0041.

## 3. Breakdowns

**By gap** (18 cells each):

| gap | mean ΔAUC | AUC wins (draw/temp) | mean ΔECE | ECE wins (draw/temp) |
|---|---|---|---|---|
| 0 | +0.00012 | 10 / 8 | −0.00097 | 12 / 6 |
| +1 | +0.00055 | 13 / 5 | −0.00057 | 12 / 6 |
| +2 | −0.00018 | 10 / 8 | +0.00138 | 5 / 13 |
| +3 | +0.00013 | 10 / 8 | +0.00226 | 6 / 12 |

ΔAUC shows no distance pattern. ΔECE changes sign with distance: the draw is slightly *better* calibrated on the two nearest days and slightly *worse* on the two farthest. For LR alone the gradient is monotone (mean ΔECE −0.00065, −0.00114, +0.00161, +0.00228 over gaps 0/+1/+2/+3; 6 cells per gap). Pooled across models it is not seed-consistent (seed-42 gap means: +0.0015, −0.0010, −0.0008, +0.0009), and at gaps +2/+3 the per-(origin × seed) means are positive in 9 of 12 cases. Treat the gradient as a pooled/LR-level pattern, not a per-seed law — the same caution the paper applies to its own day-level ECE patterns.

**By origin × model** (12 cells each): mean ΔAUC ranges from −0.00159 (origin 1 DeepFM — the one negative outlier block, driven by seed-43 cells) to +0.00111 (origin 1 DCN-v2); mean ΔECE from −0.00040 (origin 1 DCN-v2) to +0.00107 (origin 2 DCN-v2). LR is the tightest model, as elsewhere in this study: max |ΔAUC| over its 24 cells is 0.0017, against 0.0059 (DeepFM, origin 1 seed 43 gap +2) and 0.0038 (DCN-v2) — the deep models' largest cell deltas sit at their known seed-noise scale (Tier-C §3: raw-ECE seed sd up to 0.0034–0.0051 at origin 2; here the cross-seed sd of the per-cell ΔECE itself averages 0.0024, max 0.0069). The largest single ΔECE, +0.0109 (origin 2, seed 43, DCN-v2, gap +2), and LR's largest, +0.0080 (origin 1, seed 43, gap +3), are single cells inside those noise scales; no pooled quantity rests on them.

## 4. What this control establishes

1. **Row sampling does not produce the RQ1 flattery.** Holding the evaluation days, the training window, the training size (exactly, not to within 20%), the encoder protocol, and the training code fixed, changing only *which* rows of the window the model trains on — stride positions vs a uniform random draw — moves AUC by +0.0002 pooled and does not lower ECE at all (pooled +0.0005; the draw wins 35 of 72 cells, i.e. chance). The random split's +0.0135…+0.0163 AUC inflation and its 55–62% LR ECE ratio therefore cannot be attributed to random sampling of training data or to the stride sample handicapping the temporal arm. This control does not itself speak to the random arm's larger training set (the confound v15 flags in RQ1 and threat (4)); that question is closed separately, by measurement, in the equal-size retraining (results/sizematch/), which retrains the random arm at exactly the temporal arm's size and finds pure size effects of +0.0002 (LR), −0.0001 (DeepFM), +0.0002 (DCN-v2) AUC — roughly two orders of magnitude smaller than the split contrast. What remains is the protocol itself: the random split's training and test rows are drawn from the same days, and its test set spans all ten days rather than one future day. This converts v15 threat (8)'s named-but-unrun comparison into a measured one.
2. **Stride sampling is not a handicap.** A uniform draw from the same window — the natural "fairer" training set a reviewer might propose — does not beat the deterministic stride sample on the same days in either metric pooled, in any seed's pooled mean ΔECE (all three ≥ +0.0001), or at gap 0 for LR beyond ±0.0007 AUC. The temporal arm's numbers are not an artifact of its sampling pattern.
3. **A weak far-day ECE signal, stated at pooled strength only.** The draw's small ECE advantage at gaps 0/+1 reverses at +2/+3 (pooled means +0.0014/+0.0023; LR monotone). This is *consistent with* the stride sample's perfectly even temporal coverage helping calibration at distance, but the per-seed signs are inconsistent across models and the effect size is within the deep models' seed noise; it should be reported, if at all, as a pooled observation, not a finding.

## 5. What it does not establish — limits and confounds

- **Raw models only.** Tier F fits no calibrators; it says nothing about whether the draw would change any RQ3 recalibration result, harm rate, or gate behaviour. (The RQ3 size-matched control is a different control — of the *calibration slice*, not the base model's training sample — and the two should not be conflated in text.)
- **One window pair, one dataset.** The two origins share most of their days (v15 threat (2) applies unchanged); the draw pool is the same five days the stride sample covers, so the control varies sampling *within* a fixed window — it cannot speak to training-window choice, freshness of the base model, or other logs.
- **The contrast is protocol-level, not sampling-only in the strictest sense.** Each arm refits its own vocabulary encoder (the study's standing no-leakage rule), so Δ bundles the sampling change with an encoder refit on a different row subset. The vocabularies are near-identical in size (total vocab, temporal vs draw: 32,517 vs 32,537–32,586 at origin 1; 33,486 vs 33,437–33,533 at origin 2, per the arm manifests), and any encoder effect is part of what "train this protocol on a random draw" means in this pipeline — but a claim of the form "sampling alone, all else literally equal" would overstate the design.
- **Hour-of-day composition differs by construction and is not controlled.** A uniform draw reproduces the window's hour distribution in expectation; the stride sample reproduces it almost exactly (every 5th/6th row in a time-ordered file). Neither arm is fresher: both span all five days, so the control is silent on recency of base-model training data, in the same way the aged-window control found age confounded with hour of day.
- **No uncertainty machinery was rerun.** No bootstrap CIs are computed for these deltas; the noise context is the cross-seed sd of the deltas themselves (mean 0.0012 AUC / 0.0024 ECE per cell). Pooled means are reported as descriptive magnitudes against that scale, mirroring Tier-C's treatment.

## 6. Experiment A spot-check (bidding-parameter sweep)

Recomputed from `results/bidsweep/bidsweep_cells.csv` (432 rows = 9 cells × 48 deltas): every per-cell min / max / mean / n-positive in `bidsweep_summary.md` reproduces exactly (all 9 cells, all four statistics). Overall range **−0.3567% to +0.3433%** (summary/audit state: −0.357%..+0.343%); centre cell (μ = −2.2, σ = 0.9) range **−0.20704% to +0.26255%**, i.e. the paper's −0.207% to +0.263% exactly; per-cell means +0.0192 to +0.0395, positive counts 30–36 of 48. The on-disk summary matches the previously audited state; nothing differs.

## Appendix — all 72 paired cells (Δ = draw − temporal)

| origin | seed | model | gap | n_test | auc_draw | auc_temporal | d_auc | ece_draw | ece_temporal | d_ece |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 42 | dcnv2 | 0 | 3,148,694 | 0.7412 | 0.7408 | +0.0004 | 0.0079 | 0.0070 | +0.0009 |
| 1 | 42 | dcnv2 | 1 | 3,225,010 | 0.7477 | 0.7476 | +0.0002 | 0.0099 | 0.0098 | +0.0001 |
| 1 | 42 | dcnv2 | 2 | 5,287,222 | 0.7533 | 0.7522 | +0.0011 | 0.0052 | 0.0037 | +0.0015 |
| 1 | 42 | dcnv2 | 3 | 3,832,608 | 0.7417 | 0.7392 | +0.0025 | 0.0052 | 0.0058 | -0.0006 |
| 1 | 42 | deepfm | 0 | 3,148,694 | 0.7421 | 0.7431 | -0.0010 | 0.0074 | 0.0069 | +0.0005 |
| 1 | 42 | deepfm | 1 | 3,225,010 | 0.7487 | 0.7521 | -0.0034 | 0.0062 | 0.0092 | -0.0030 |
| 1 | 42 | deepfm | 2 | 5,287,222 | 0.7521 | 0.7569 | -0.0048 | 0.0078 | 0.0078 | +0.0000 |
| 1 | 42 | deepfm | 3 | 3,832,608 | 0.7418 | 0.7418 | +0.0000 | 0.0117 | 0.0061 | +0.0055 |
| 1 | 42 | lr | 0 | 3,148,694 | 0.7321 | 0.7320 | +0.0000 | 0.0088 | 0.0092 | -0.0003 |
| 1 | 42 | lr | 1 | 3,225,010 | 0.7389 | 0.7384 | +0.0005 | 0.0098 | 0.0118 | -0.0021 |
| 1 | 42 | lr | 2 | 5,287,222 | 0.7497 | 0.7497 | -0.0001 | 0.0085 | 0.0084 | +0.0001 |
| 1 | 42 | lr | 3 | 3,832,608 | 0.7349 | 0.7352 | -0.0003 | 0.0087 | 0.0054 | +0.0033 |
| 1 | 43 | dcnv2 | 0 | 3,148,694 | 0.7411 | 0.7414 | -0.0003 | 0.0093 | 0.0102 | -0.0009 |
| 1 | 43 | dcnv2 | 1 | 3,225,010 | 0.7488 | 0.7485 | +0.0003 | 0.0112 | 0.0078 | +0.0034 |
| 1 | 43 | dcnv2 | 2 | 5,287,222 | 0.7535 | 0.7522 | +0.0013 | 0.0077 | 0.0079 | -0.0002 |
| 1 | 43 | dcnv2 | 3 | 3,832,608 | 0.7408 | 0.7401 | +0.0007 | 0.0099 | 0.0116 | -0.0018 |
| 1 | 43 | deepfm | 0 | 3,148,694 | 0.7413 | 0.7431 | -0.0018 | 0.0064 | 0.0095 | -0.0031 |
| 1 | 43 | deepfm | 1 | 3,225,010 | 0.7478 | 0.7526 | -0.0048 | 0.0089 | 0.0064 | +0.0025 |
| 1 | 43 | deepfm | 2 | 5,287,222 | 0.7512 | 0.7571 | -0.0059 | 0.0101 | 0.0080 | +0.0021 |
| 1 | 43 | deepfm | 3 | 3,832,608 | 0.7395 | 0.7419 | -0.0025 | 0.0115 | 0.0095 | +0.0020 |
| 1 | 43 | lr | 0 | 3,148,694 | 0.7315 | 0.7320 | -0.0005 | 0.0088 | 0.0091 | -0.0003 |
| 1 | 43 | lr | 1 | 3,225,010 | 0.7390 | 0.7385 | +0.0005 | 0.0093 | 0.0115 | -0.0022 |
| 1 | 43 | lr | 2 | 5,287,222 | 0.7489 | 0.7498 | -0.0009 | 0.0101 | 0.0081 | +0.0020 |
| 1 | 43 | lr | 3 | 3,832,608 | 0.7343 | 0.7352 | -0.0009 | 0.0139 | 0.0059 | +0.0080 |
| 1 | 44 | dcnv2 | 0 | 3,148,694 | 0.7413 | 0.7393 | +0.0020 | 0.0089 | 0.0082 | +0.0007 |
| 1 | 44 | dcnv2 | 1 | 3,225,010 | 0.7494 | 0.7465 | +0.0029 | 0.0107 | 0.0111 | -0.0004 |
| 1 | 44 | dcnv2 | 2 | 5,287,222 | 0.7542 | 0.7532 | +0.0009 | 0.0030 | 0.0068 | -0.0038 |
| 1 | 44 | dcnv2 | 3 | 3,832,608 | 0.7417 | 0.7405 | +0.0012 | 0.0067 | 0.0105 | -0.0038 |
| 1 | 44 | deepfm | 0 | 3,148,694 | 0.7437 | 0.7412 | +0.0024 | 0.0040 | 0.0057 | -0.0016 |
| 1 | 44 | deepfm | 1 | 3,225,010 | 0.7523 | 0.7497 | +0.0026 | 0.0055 | 0.0073 | -0.0018 |
| 1 | 44 | deepfm | 2 | 5,287,222 | 0.7528 | 0.7538 | -0.0010 | 0.0068 | 0.0052 | +0.0015 |
| 1 | 44 | deepfm | 3 | 3,832,608 | 0.7420 | 0.7411 | +0.0009 | 0.0105 | 0.0054 | +0.0051 |
| 1 | 44 | lr | 0 | 3,148,694 | 0.7321 | 0.7320 | +0.0001 | 0.0091 | 0.0091 | +0.0000 |
| 1 | 44 | lr | 1 | 3,225,010 | 0.7389 | 0.7385 | +0.0004 | 0.0101 | 0.0116 | -0.0015 |
| 1 | 44 | lr | 2 | 5,287,222 | 0.7509 | 0.7497 | +0.0011 | 0.0074 | 0.0081 | -0.0008 |
| 1 | 44 | lr | 3 | 3,832,608 | 0.7357 | 0.7352 | +0.0005 | 0.0087 | 0.0059 | +0.0029 |
| 2 | 42 | dcnv2 | 0 | 2,575,774 | 0.7547 | 0.7558 | -0.0011 | 0.0080 | 0.0044 | +0.0036 |
| 2 | 42 | dcnv2 | 1 | 5,287,222 | 0.7563 | 0.7573 | -0.0010 | 0.0059 | 0.0064 | -0.0006 |
| 2 | 42 | dcnv2 | 2 | 3,832,608 | 0.7441 | 0.7440 | +0.0001 | 0.0076 | 0.0101 | -0.0025 |
| 2 | 42 | dcnv2 | 3 | 4,218,938 | 0.7369 | 0.7382 | -0.0013 | 0.0083 | 0.0081 | +0.0002 |
| 2 | 42 | deepfm | 0 | 2,575,774 | 0.7561 | 0.7562 | -0.0001 | 0.0092 | 0.0040 | +0.0052 |
| 2 | 42 | deepfm | 1 | 5,287,222 | 0.7575 | 0.7578 | -0.0003 | 0.0071 | 0.0067 | +0.0004 |
| 2 | 42 | deepfm | 2 | 3,832,608 | 0.7454 | 0.7443 | +0.0012 | 0.0043 | 0.0116 | -0.0074 |
| 2 | 42 | deepfm | 3 | 4,218,938 | 0.7367 | 0.7381 | -0.0014 | 0.0080 | 0.0103 | -0.0022 |
| 2 | 42 | lr | 0 | 2,575,774 | 0.7464 | 0.7457 | +0.0007 | 0.0085 | 0.0097 | -0.0012 |
| 2 | 42 | lr | 1 | 5,287,222 | 0.7534 | 0.7517 | +0.0017 | 0.0093 | 0.0101 | -0.0008 |
| 2 | 42 | lr | 2 | 3,832,608 | 0.7372 | 0.7368 | +0.0004 | 0.0104 | 0.0072 | +0.0032 |
| 2 | 42 | lr | 3 | 4,218,938 | 0.7291 | 0.7277 | +0.0014 | 0.0101 | 0.0108 | -0.0007 |
| 2 | 43 | dcnv2 | 0 | 2,575,774 | 0.7559 | 0.7564 | -0.0005 | 0.0072 | 0.0146 | -0.0074 |
| 2 | 43 | dcnv2 | 1 | 5,287,222 | 0.7593 | 0.7576 | +0.0017 | 0.0133 | 0.0129 | +0.0004 |
| 2 | 43 | dcnv2 | 2 | 3,832,608 | 0.7449 | 0.7449 | -0.0000 | 0.0186 | 0.0077 | +0.0109 |
| 2 | 43 | dcnv2 | 3 | 4,218,938 | 0.7392 | 0.7393 | -0.0001 | 0.0155 | 0.0119 | +0.0036 |
| 2 | 43 | deepfm | 0 | 2,575,774 | 0.7564 | 0.7557 | +0.0007 | 0.0070 | 0.0098 | -0.0028 |
| 2 | 43 | deepfm | 1 | 5,287,222 | 0.7591 | 0.7558 | +0.0033 | 0.0103 | 0.0103 | -0.0000 |
| 2 | 43 | deepfm | 2 | 3,832,608 | 0.7453 | 0.7442 | +0.0011 | 0.0096 | 0.0062 | +0.0034 |
| 2 | 43 | deepfm | 3 | 4,218,938 | 0.7377 | 0.7368 | +0.0009 | 0.0145 | 0.0077 | +0.0068 |
| 2 | 43 | lr | 0 | 2,575,774 | 0.7464 | 0.7459 | +0.0005 | 0.0082 | 0.0093 | -0.0011 |
| 2 | 43 | lr | 1 | 5,287,222 | 0.7517 | 0.7520 | -0.0003 | 0.0098 | 0.0095 | +0.0003 |
| 2 | 43 | lr | 2 | 3,832,608 | 0.7370 | 0.7371 | -0.0001 | 0.0104 | 0.0074 | +0.0030 |
| 2 | 43 | lr | 3 | 4,218,938 | 0.7285 | 0.7281 | +0.0003 | 0.0111 | 0.0107 | +0.0004 |
| 2 | 44 | dcnv2 | 0 | 2,575,774 | 0.7541 | 0.7533 | +0.0008 | 0.0059 | 0.0088 | -0.0029 |
| 2 | 44 | dcnv2 | 1 | 5,287,222 | 0.7572 | 0.7534 | +0.0038 | 0.0060 | 0.0071 | -0.0011 |
| 2 | 44 | dcnv2 | 2 | 3,832,608 | 0.7440 | 0.7428 | +0.0012 | 0.0135 | 0.0092 | +0.0043 |
| 2 | 44 | dcnv2 | 3 | 4,218,938 | 0.7376 | 0.7365 | +0.0011 | 0.0120 | 0.0078 | +0.0042 |
| 2 | 44 | deepfm | 0 | 2,575,774 | 0.7548 | 0.7555 | -0.0007 | 0.0041 | 0.0101 | -0.0060 |
| 2 | 44 | deepfm | 1 | 5,287,222 | 0.7574 | 0.7559 | +0.0015 | 0.0049 | 0.0081 | -0.0032 |
| 2 | 44 | deepfm | 2 | 3,832,608 | 0.7449 | 0.7436 | +0.0013 | 0.0111 | 0.0057 | +0.0054 |
| 2 | 44 | deepfm | 3 | 4,218,938 | 0.7365 | 0.7371 | -0.0005 | 0.0126 | 0.0045 | +0.0081 |
| 2 | 44 | lr | 0 | 2,575,774 | 0.7462 | 0.7459 | +0.0004 | 0.0084 | 0.0095 | -0.0010 |
| 2 | 44 | lr | 1 | 5,287,222 | 0.7521 | 0.7520 | +0.0002 | 0.0093 | 0.0099 | -0.0006 |
| 2 | 44 | lr | 2 | 3,832,608 | 0.7368 | 0.7371 | -0.0003 | 0.0093 | 0.0072 | +0.0021 |
| 2 | 44 | lr | 3 | 4,218,938 | 0.7278 | 0.7280 | -0.0002 | 0.0105 | 0.0107 | -0.0002 |
