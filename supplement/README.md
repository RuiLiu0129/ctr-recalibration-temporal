# Audit supplement — "How Much Fresh Data Should CTR Recalibration Use?" (BigData workshop submission, v20)

Cell-level, **unrounded** data behind the paper's tables and headline statistics, so that
any number in the PDF can be recomputed exactly. All files are consolidated from the
experiment project's existing result files; no new experiments were run for this supplement.
Project root referenced below: `avazu-temporal-calibration/results/`.

Conventions: a recalibration **harms** a cell iff `delta_ece = cal_ece − raw_ece > 0`.
Models: `lr`, `deepfm`, `dcnv2`. Gaps are test-bucket offsets in days from the training
window. Seed is 42 unless a file says otherwise.

## Files

| File | Contents | Source in project | Paper claim(s) it supports |
|---|---|---|---|
| `grid48_unrounded.csv` | All 48 cells (2 origins × 3 models × 4 gaps × 2 calibrators): `raw_ece`, `cal_ece`, `delta_ece`, raw **and calibrated** AUC and log loss, raw/cal signed bias, `n_test`, `n_calib`. | `o1_repro/metrics_temporal.csv`, `origin2/metrics_temporal.csv` (per-variant rows pivoted; `metrics_temporal.csv` logs AUC/log loss/ECE for raw, Platt and isotonic variants, so calibrated AUCs are present — isotonic/Platt are monotone maps, hence calibrated AUC equals raw AUC except for isotonic tie effects, exactly as logged) | Table IV grid; "16/48 harmful at 6 h, seed 42"; "largest reduction ≈ 61%", "largest increase ≈ +107%"; "calibrated AUC changes by at most 0.0002" |
| `slice_target_bias.csv` | Per model–bucket (24 rows): fitting-slice size, slice observed CTR, slice mean prediction, **slice signed bias**, slice ECE; and the target bucket's raw signed bias (Table I quantities). | `o1_repro/slice_stats_temporal.csv`, `origin2/slice_stats_temporal.csv`, `*/metrics_temporal.csv` | §IV-C mechanism claims; the slice-bias sign rule (40/48) and magnitude rule (42/48) — both recomputable cell-by-cell from this file plus `grid48_unrounded.csv` harm flags |
| `gate_cells.csv` | Per model–bucket: slice ECE, slice/target bias, the τ_slice = 0.0061 gate firing (fire iff slice ECE ≥ τ) and the resulting gated ECE; the coarser τ = 0.0065 replay firing; the retrospective target-ECE rule's harm prediction (`raw_ece ≤ 0.0065`) vs actual Platt/isotonic harm. | Consolidated from `*/slice_stats_temporal.csv` and `tierb/cells48.csv` | §IV-C gate paragraph: gated means 0.0063 (origin 1) / 0.0082 (origin 2) reproduce `gate_eval_summary.csv`; the 0.0065 replay fires on 6 of 12 origin-1 buckets |
| `gate_eval_summary.csv` | Slice-ECE gate means (never / always-isotonic / gated) per origin. | Verbatim copy of `tierb/gate_eval.csv` | §IV-C (i)–(ii) |
| `rule_scores_summary.csv` | Accuracy of the three retrospective harm rules (raw-ECE τ = 0.0065, gap = +3, weekday-absent), fitted on origin 1, frozen on origin 2. | Verbatim copy of `tierb/rule_scores.csv` | §IV-C "simple rules degrade out-of-origin" (0.917 → 0.667 / 0.583; weekday 0.333 pooled) |
| `slice_target_corr.csv` | Pearson/Spearman correlation between slice ECE and target-bucket raw ECE per origin. | Verbatim copy of `tierb/slice_target_corr.csv` | §IV-C (iii): r ≈ 0.34 / 0.38 per origin, ≈ 0.39 pooled |
| `slice_bias_gates.csv` | The four bias-aware slice gates (magnitude, sign-only, ECE − \|bias\|): fitted τ, per-origin never/always/gated means, harm accuracy, firing counts. | Verbatim copy of `tierc/slice_bias_gates.csv` | §IV-C closing paragraph: gains of 0.0002–0.0004 over always-isotonic, within seed noise; no gated mean beats always-Platt (0.0075) |
| `seed_o1_ece.csv` | Origin 1 per-seed ECE for raw/Platt/isotonic (seeds 42/43/44), means and SDs. | Verbatim copy of `tierb/seed_ece.csv` | Seed-stability statements in §IV-A/§IV-B; RQ2 seed means |
| `seed_o1_delta_flags.csv` | Origin 1 per-seed ΔECE per cell, sign-flip and within-seed-noise flags. | Verbatim copy of `tierb/seed_flags.csv` | "11 of 24 origin-1 cells within seed noise" |
| `seed_o2_delta_ece.csv` | Origin 2 per-seed ΔECE per cell with SDs and noise flags. | Verbatim copy of `tierc/seed_o2_ece.csv` | Origin-2 seed replication (14/24 within noise) |
| `seed_random_arm.csv` | Random arm per-seed AUC/log loss/ECE by variant. | Verbatim copy of `tierc/seed_random_arm.csv` | RQ1: AUC inflation seed-stable; gap-0 ECE ratios per seed (LR 0.55/0.59/0.62; DeepFM 0.55/0.61/0.89; DCN-v2 0.74/1.04/0.58) |
| `seed_harm_counts.csv` | Harm-cell counts by arm × seed × calibrator, computed from the four seed files above. | Derived (this supplement) | Abstract: "14–17 of 48 applications across seeds" (per-seed totals across both origins: 16, 14, 17) |
| `delta_ece_bootstrap.csv` | Paired block-bootstrap 95% CIs for ΔECE, all 48 cells (200 blocks, 10,000 resamples). | Verbatim copy of `tierc/delta_ece_bootstrap.csv` | "43 of 48 intervals exclude zero" |
| `rq4_bootstrap.csv` | Paired block-bootstrap profit deltas per cell (absolute and per-1,000-impressions), CIs, `ci_excludes_0`. | Verbatim copy of `tierb/rq4_bootstrap.csv` | Table VI asterisks: 24 of 48 intervals exclude zero; range −0.21%…+0.26% |
| `rq4_binned_oracle.csv` | Per-cell profits of raw / Platt / isotonic / **binned calibration oracle** / label oracle. | Verbatim copy of `tierb/rq4_calib_oracle.csv` | §IV-E: binned oracle below raw in 24/24 point estimates (significant in 23), median −0.37 per 1,000; label oracle 1.75×–1.97× |
| `rq4_isotonic_oracle.csv` | Per-cell isotonic test-day oracle profit, delta vs raw, bootstrap CI. | Verbatim copy of `tierc/rq4_isotonic_oracle.csv` | §IV-E: isotonic oracle beats raw in 24/24 (all significant), median +0.147 per 1,000 |
| `sweep_seed42_delta_ece.csv` | Slice-length sweep, seed 42: ΔECE per cell for slice lengths 1/3/6/12/24 h, both origins, with slice sizes (`n_calib`). Note: at 24 h the gap-0 bucket has no impressions left to evaluate, so that grid has 36 cells, not 48. | Verbatim copy of `tierc/sweep_delta_ece.csv` | Table V / §IV-D dose–response |
| `sweep_harm_counts.csv` | Harm counts per slice length, split into the **fixed evaluation set** (gaps +1…+3, identical cells at every length: 36 cells) vs **all cells**. | Derived from `sweep_seed42_delta_ece.csv` | "13/36 → 4/36 on fixed evaluation sets" (6 h → 12 h). The all-cells counts are 16/48 → 7/48 and must not be substituted for the fixed-set numbers |
| `reproduce_sign_corr.py` | Script: reads `delta_ece_bootstrap.csv`, `rq4_bootstrap.csv`, `rq4_binned_oracle.csv` from this directory and prints the sign-agreement count and the three Pearson correlations. | This supplement | §IV-E: "the sign of the profit delta tracks the sign of the ECE change in 40 of 48 cells"; r = −0.69 pooled, −0.77 / −0.78 per origin |

## Tier-D additions

Added 2026-10-23. Unlike the files above, these report **new experiment runs**
(Tier-D, 2026-10-22/23): the slice-length sweep replicated at seeds 43 and 44,
and the mechanism control experiment (seed 42). Together with
`sweep_seed42_delta_ece.csv` (seed 42) they underlie the three-seed sweep
table (Table V) and the size-matched control. Narrative analyses are in the repository at
`docs/sweep_seeds_analysis.md` and `docs/mechanism_analysis.md`.

| File | Contents | Source in project | Paper claim(s) it supports |
|---|---|---|---|
| `sweep_seeds_cells.csv` | Slice-length sweep at seeds 43/44: per-cell metrics (raw/Platt/isotonic AUC, log loss, ECE, signed bias, `n_test`, `n_calib`) for `calib_hours` 1/3/12/24 at both origins, with `seed`, `calib_hours`, `origin` prepended (540 rows). At 24 h the gap-0 bucket does not exist by construction, so those grids have 27 rows per origin instead of 36. The 6 h points are the Tier-B/C seed runs (see the seed files above). | Consolidated from `sweep_seeds/seed{43,44}/hours_{1,3,12,24}/origin{1,2}/metrics_temporal.csv` | Table V / §IV-D three-seed sweep: the 12 h harm minimum replicates in all 6 seed × origin combinations |
| `sweep_seeds_slice_stats.csv` | Fitting-slice statistics (slice size, observed CTR, mean prediction, signed bias, AUC, log loss, ECE) for the same sweep runs (180 rows). | Consolidated from `sweep_seeds/seed{43,44}/hours_{h}/origin{o}/slice_stats_temporal.csv` | Slice-size axis of the sweep table |
| `sweep_seeds_timings.csv` | Measured per-cell calibrator fit/apply wall-clock times (seconds) for the same sweep runs (360 rows). | Consolidated from `sweep_seeds/seed{43,44}/hours_{h}/origin{o}/timings_temporal.csv` | Measured refit costs by slice length, Table III (with the same-run control timings in `mechanism_cells.csv`) | 
| `mechanism_cells.csv` | Mechanism control, seed 42: per model × gap × calibrator, ΔECE under three fitting conditions — (a) `full12`: the full 12 h slice; (b) `hour1`: the actual 1 h slice; (c) `sub`: K=5 uniform subsamples of the 12 h slice at exactly the 1 h slice's impression count — with `n_fit`, fit/apply times, and the subsample seed for each row (336 rows). | Concatenation of `mechanism_slice/mechanism_o1.csv` and `mechanism_slice/mechanism_o2.csv` | Size-matched control (§IV-C): the size-matched 12 h subsample recovers 87% of the full 12 h benefit relative to the 1 h slice, (sub − hour1)/(full12 − hour1), on the fixed gaps +1…+3 cells the paper reports (0.80–0.99 per origin × calibrator; 88% and 0.68–0.93 if the gap-0 cells are included) and matches its harm count (4 of 36, against 15 for the true 1 h slice, whose mean ΔECE is −0.0001 on these cells and +0.0007 with gap 0) — impression count alone does not explain the improvement; span also changes hour coverage and data age |
| `mechanism_o1_validation.json`, `mechanism_o2_validation.json` | Validation of the mechanism pipeline against the independent sweep runs (full-12 h ΔECE agrees to ~1e-16, 1 h at gaps 1–3 to ~1e-13), the elementwise 1 h ⊂ 12 h subset check, and the subsample seed registry. | Verbatim copies of `mechanism_slice/mechanism_o{1,2}_validation.json` | Internal validity of the mechanism experiment |


## One-command reproduction (v20)

`python3 reproduce_paper_numbers.py` (pandas, scipy) recomputes 87 quantitative claims of the paper, including the Tier-E controls (size-matched control at three seeds, aged-window control, and the hour-composition statements of §IV-D) and the Tier-F additions (matched-sampling control, equal-size retraining of the random arm, price sweep), from the files above, prints each next to the value printed in the paper, and exits with the number of mismatches (0 for v20). It also writes two derived tables the paper summarises:

| File | Contents |
|---|---|
| `sweep_fixed_cells_three_seeds.csv` | ΔECE and harm flag per seed × origin × model × gap (+1…+3) × calibrator × slice length (1/3/6/12/24 h) — the cells behind Table V |
| `sweep_harm_by_seed_origin.csv` | Harm counts per seed × origin × slice length — supports "12 h has the fewest harms in all six seed × origin combinations" |

## The sign-agreement rule, stated explicitly

A cell **agrees** iff ΔECE and Δprofit have **opposite signs** (calibration improves →
profit rises, or calibration worsens → profit falls). Δprofit is the percentage change
relative to raw profit, as printed in Table VI. A delta of exactly 0.0 would count as
non-agreement; **no unrounded ΔECE or Δprofit in the data is exactly zero**, so the rule
never binds. In earlier PDF versions three deltas printed as +0.00%; the paper now
prints them to three decimals (+0.001, +0.002, +0.004), so the 40/48 count can be read
off the table (38 + 2 agree, 7 + 1 disagree). Reproduction output:

```
sign agreements: 40 of 48
Pearson r (pooled): -0.6866
Pearson r (origin 1): -0.7695
Pearson r (origin 2): -0.7830
```

## Run logs and environment

Added for the v9 submission package. The Table III training times were the one
set of numbers in the paper whose provenance was not yet in this supplement;
the raw run logs are now included verbatim.

| File | Contents | Paper claim(s) it supports |
|---|---|---|
| `logs/tierb_o1repro.log` | Raw log of the seed-42 origin-1 headline run (temporal arm). Per-model train times: line 6 `lr: trained in 10.5s`, line 11 `deepfm: trained in 205.7s`, line 16 `dcnv2: trained in 599.8s`; line 21 `train+eval done in 1290.6s`. | Table III origin-1 training column (10.5 / 205.7 / 599.8 s) and the 1,291 s arm total (the printed total is 1290.6 s rounded) |
| `logs/tierb_o2.log` | Raw log of the seed-42 origin-2 headline run (temporal arm). Per-model train times: line 6 `lr: trained in 13.1s`, line 11 `deepfm: trained in 257.6s`, line 16 `dcnv2: trained in 703.0s`; line 21 `train+eval done in 1457.4s`. | Table III origin-2 training column (13.1 / 257.6 / 703.0 s) and the 1,457 s arm total (1457.4 s rounded) |
| `ENVIRONMENT.md` | Exact library versions printed in the paper and the remaining environment pins, how they were obtained (read from the project venv that ran the experiments), and the hardware line (2-vCPU VM, AMD EPYC 9D25 host). The versions are pinned by `requirements.txt` in the public repository (github.com/RuiLiu0129/ctr-recalibration-temporal). | §IV-C software/hardware specification; the environment behind every timing in Table III |

## Tier-E additions (v10)

Added 2026-10-23. Two follow-up controls reported in the v10 text:

| File | Contents | Paper claim(s) it supports |
|---|---|---|
| `tiere/mechanism_seeds/seed43/mechanism_o{1,2}.csv`, `tiere/mechanism_seeds/seed44/mechanism_o{1,2}.csv` | Size-matched control replicated at seeds 43/44: per-cell ΔECE for the full 12 h slice, the true 1 h slice, and K=5 size-matched subsamples (seed 42 cells are in `mechanism_cells.csv`). | §IV-C / abstract: three-seed pooled recovery 88% (0.8745 / 0.8791 / 0.8794 by seed; 0.72–1.00 across seed×origin×calibrator); subsample harms 4/7/4 of 36 vs full-slice 4/5/3 and true-1h 15/15/18 |
| `tiere/recency/recency_o{1,2}.csv` | Aged-window (recency) control, seed 42: per-cell ΔECE for 6 h windows aged 0/6/12/18 h plus the full-24 h reference, gaps +1..+3, both origins. | §IV-D / threats (5): age means −0.0003 / −0.0018 / +0.0004 / +0.0004 (harms 13/8/19/15 of 36), full-24 h −0.0019 (9/36); not monotone in age |
| `tiere/mechanism_seeds_analysis.md`, `tiere/recency_analysis.md` | Full Tier-E analyses, with per-origin × calibrator tables and validation summaries. | Same as above |
| `tierf/matched_sampling_cells.csv` | Matched-sampling control, three seeds, both origins: base models retrained on a uniform draw (without replacement) from the temporal training window at the temporal arm's exact size, scored raw on the same four target days; 72 cells with paired temporal reference. | §IV-A: pooled dAUC +0.00015, dECE +0.00053, ahead on ECE in 35/72; gap-0 dAUC +0.00012 pooled, +0.00015 at origin 1 (9 cells) |
| `tierf/equal_size_random_cells.csv`, `tierf/equal_size_contrasts.csv` | Random arm retrained at origin 1's size (3,338,217 training rows; 70/10/20 pool scaled with it, 953,777 test rows), three seeds: raw/Platt/isotonic metrics, and per-seed contrasts against the temporal gap-0 bucket (`a_sizefree_split`) and against the capped random arm (`b_pure_size`). | §IV-A / threats (4), (8): gap-0 AUC contrast +0.0140/+0.0150/+0.0154 vs capped +0.0137/+0.0151/+0.0152; difference +0.0002/−0.0001/+0.0002; LR ECE ratio 0.61–0.64 (capped 0.55/0.59/0.62) |
| `tierf/bid_price_sweep.csv` | Bidding simulation over a 3×3 lognormal price grid (μ ∈ {−2.6, −2.2, −1.8}, σ ∈ {0.6, 0.9, 1.2}), 48 recalibration profit deltas per setting, seed 42. | §IV-E / threats (7): all deltas −0.357% to +0.343% (within ±0.36%); per-setting means +0.019% to +0.040%; headline setting reproduces −0.207% to +0.263% |
