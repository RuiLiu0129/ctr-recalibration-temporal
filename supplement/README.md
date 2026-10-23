# Audit supplement — "Short-Slice Recalibration Can Hurt" (BigData workshop submission, v5)

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
| `grid48_unrounded.csv` | All 48 cells (2 origins × 3 models × 4 gaps × 2 calibrators): `raw_ece`, `cal_ece`, `delta_ece`, raw **and calibrated** AUC and log loss, raw/cal signed bias, `n_test`, `n_calib`. | `o1_repro/metrics_temporal.csv`, `origin2/metrics_temporal.csv` (per-variant rows pivoted; `metrics_temporal.csv` logs AUC/log loss/ECE for raw, Platt and isotonic variants, so calibrated AUCs are present — isotonic/Platt are monotone maps, hence calibrated AUC equals raw AUC except for isotonic tie effects, exactly as logged) | Table III grid; "16/48 harmful at 6 h, seed 42"; "largest reduction ≈ 61%", "largest increase ≈ +107%"; "calibrated AUC changes by at most 0.0002" |
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
| `rq4_bootstrap.csv` | Paired block-bootstrap profit deltas per cell (absolute and per-1,000-impressions), CIs, `ci_excludes_0`. | Verbatim copy of `tierb/rq4_bootstrap.csv` | Table V asterisks: 24 of 48 intervals exclude zero; range −0.21%…+0.26% |
| `rq4_binned_oracle.csv` | Per-cell profits of raw / Platt / isotonic / **binned calibration oracle** / label oracle. | Verbatim copy of `tierb/rq4_calib_oracle.csv` | §IV-E: binned oracle below raw in 24/24 point estimates (significant in 23), median −0.37 per 1,000; label oracle 1.75×–1.97× |
| `rq4_isotonic_oracle.csv` | Per-cell isotonic test-day oracle profit, delta vs raw, bootstrap CI. | Verbatim copy of `tierc/rq4_isotonic_oracle.csv` | §IV-E: isotonic oracle beats raw in 24/24 (all significant), median +0.147 per 1,000 |
| `sweep_seed42_delta_ece.csv` | Slice-length sweep, seed 42: ΔECE per cell for slice lengths 1/3/6/12/24 h, both origins, with slice sizes (`n_calib`). Note: at 24 h the gap-0 bucket has no impressions left to evaluate, so that grid has 36 cells, not 48. | Verbatim copy of `tierc/sweep_delta_ece.csv` | Table IV / §IV-D dose–response |
| `sweep_harm_counts.csv` | Harm counts per slice length, split into the **fixed evaluation set** (gaps +1…+3, identical cells at every length: 36 cells) vs **all cells**. | Derived from `sweep_seed42_delta_ece.csv` | "13/36 → 4/36 on fixed evaluation sets" (6 h → 12 h). The all-cells counts are 16/48 → 7/48 and must not be substituted for the fixed-set numbers |
| `reproduce_sign_corr.py` | Script: reads `delta_ece_bootstrap.csv`, `rq4_bootstrap.csv`, `rq4_binned_oracle.csv` from this directory and prints the sign-agreement count and the three Pearson correlations. | This supplement | §IV-E: "the sign of the profit delta tracks the sign of the ECE change in 40 of 48 cells"; r = −0.69 pooled, −0.77 / −0.78 per origin |

## The sign-agreement rule, stated explicitly

A cell **agrees** iff ΔECE and Δprofit have **opposite signs** (calibration improves →
profit rises, or calibration worsens → profit falls). Δprofit is the percentage change
relative to raw profit, as printed in Table V. A delta of exactly 0.0 would count as
non-agreement; **no unrounded ΔECE or Δprofit in the data is exactly zero**, so the rule
never binds — the ambiguity exists only in the PDF's rounded table, where two deltas
print as +0.00%. Reproduction output:

```
sign agreements: 40 of 48
Pearson r (pooled): -0.6866
Pearson r (origin 1): -0.7695
Pearson r (origin 2): -0.7830
```
