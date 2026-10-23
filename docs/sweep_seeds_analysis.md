# Tier-D Experiment 1 — slice-length sweep at seeds 43/44 (2026-10-22)

Same cells and conventions as Tier-C section 4 (seed 42, recomputed here in the same code path; hours_6 for seeds 43/44 is the Tier-B/C seed run). dECE = ECE(calibrated) - ECE(raw); harm = dECE > 0. At 24 h the gap-0 bucket does not exist by construction (18 of 24 cells, 9 per calibrator).

## Pooled (both calibrators): mean dECE and harm counts

| seed | origin | calib_hours | mean_delta_ece | n_harm | n_cells |
|---|---|---|---|---|---|
| 42 | 1 | 1 | +0.0018 | 13 | 24 |
| 42 | 1 | 3 | +0.0002 | 14 | 24 |
| 42 | 1 | 6 | -0.0007 | 8 | 24 |
| 42 | 1 | 12 | -0.0021 | 3 | 24 |
| 42 | 1 | 24 | -0.0015 | 6 | 18 |
| 42 | 2 | 1 | -0.0002 | 12 | 24 |
| 42 | 2 | 3 | +0.0002 | 11 | 24 |
| 42 | 2 | 6 | -0.0005 | 8 | 24 |
| 42 | 2 | 12 | -0.0019 | 4 | 24 |
| 42 | 2 | 24 | -0.0023 | 3 | 18 |
| 43 | 1 | 1 | +0.0007 | 12 | 24 |
| 43 | 1 | 3 | -0.0014 | 6 | 24 |
| 43 | 1 | 6 | -0.0023 | 3 | 24 |
| 43 | 1 | 12 | -0.0037 | 1 | 24 |
| 43 | 1 | 24 | -0.0029 | 5 | 18 |
| 43 | 2 | 1 | -0.0002 | 10 | 24 |
| 43 | 2 | 3 | -0.0007 | 13 | 24 |
| 43 | 2 | 6 | -0.0015 | 11 | 24 |
| 43 | 2 | 12 | -0.0026 | 5 | 24 |
| 43 | 2 | 24 | -0.0023 | 6 | 18 |
| 44 | 1 | 1 | +0.0005 | 11 | 24 |
| 44 | 1 | 3 | -0.0009 | 8 | 24 |
| 44 | 1 | 6 | -0.0015 | 4 | 24 |
| 44 | 1 | 12 | -0.0024 | 2 | 24 |
| 44 | 1 | 24 | -0.0021 | 6 | 18 |
| 44 | 2 | 1 | +0.0012 | 13 | 24 |
| 44 | 2 | 3 | +0.0017 | 17 | 24 |
| 44 | 2 | 6 | +0.0003 | 13 | 24 |
| 44 | 2 | 12 | -0.0016 | 3 | 24 |
| 44 | 2 | 24 | -0.0013 | 5 | 18 |

## Per calibrator: mean dECE, harm/n

- seed 42, origin 1, platt: 1h: +0.0018, harm 6/12; 3h: +0.0009, harm 7/12; 6h: -0.0002, harm 3/12; 12h: -0.0010, harm 2/12; 24h: -0.0004, harm 3/9.
- seed 42, origin 1, isotonic: 1h: +0.0018, harm 7/12; 3h: -0.0006, harm 7/12; 6h: -0.0012, harm 5/12; 12h: -0.0031, harm 1/12; 24h: -0.0025, harm 3/9.
- seed 42, origin 2, platt: 1h: -0.0015, harm 3/12; 3h: -0.0000, harm 4/12; 6h: -0.0008, harm 3/12; 12h: -0.0019, harm 1/12; 24h: -0.0014, harm 2/9.
- seed 42, origin 2, isotonic: 1h: +0.0010, harm 9/12; 3h: +0.0004, harm 7/12; 6h: -0.0002, harm 5/12; 12h: -0.0019, harm 3/12; 24h: -0.0033, harm 1/9.
- seed 43, origin 1, platt: 1h: +0.0008, harm 6/12; 3h: -0.0008, harm 3/12; 6h: -0.0019, harm 1/12; 12h: -0.0029, harm 1/12; 24h: -0.0020, harm 4/9.
- seed 43, origin 1, isotonic: 1h: +0.0006, harm 6/12; 3h: -0.0020, harm 3/12; 6h: -0.0028, harm 2/12; 12h: -0.0044, harm 0/12; 24h: -0.0037, harm 1/9.
- seed 43, origin 2, platt: 1h: -0.0012, harm 4/12; 3h: -0.0010, harm 5/12; 6h: -0.0014, harm 5/12; 12h: -0.0020, harm 3/12; 24h: -0.0011, harm 5/9.
- seed 43, origin 2, isotonic: 1h: +0.0008, harm 6/12; 3h: -0.0005, harm 8/12; 6h: -0.0016, harm 6/12; 12h: -0.0032, harm 2/12; 24h: -0.0035, harm 1/9.
- seed 44, origin 1, platt: 1h: +0.0004, harm 5/12; 3h: -0.0004, harm 4/12; 6h: -0.0011, harm 2/12; 12h: -0.0015, harm 1/12; 24h: -0.0010, harm 4/9.
- seed 44, origin 1, isotonic: 1h: +0.0007, harm 6/12; 3h: -0.0013, harm 4/12; 6h: -0.0018, harm 2/12; 12h: -0.0032, harm 1/12; 24h: -0.0031, harm 2/9.
- seed 44, origin 2, platt: 1h: +0.0007, harm 6/12; 3h: +0.0018, harm 8/12; 6h: +0.0005, harm 6/12; 12h: -0.0014, harm 2/12; 24h: -0.0003, harm 4/9.
- seed 44, origin 2, isotonic: 1h: +0.0017, harm 7/12; 3h: +0.0017, harm 9/12; 6h: +0.0002, harm 7/12; 12h: -0.0017, harm 1/12; 24h: -0.0024, harm 1/9.

## Canonical cells (dECE by slice length)

- O1 lr gap3 platt, seed 42: 1h +0.0072, 3h +0.0060, 6h +0.0058, 12h +0.0001, 24h +0.0022.
- O1 lr gap3 platt, seed 43: 1h +0.0069, 3h +0.0056, 6h +0.0055, 12h -0.0002, 24h +0.0019.
- O1 lr gap3 platt, seed 44: 1h +0.0069, 3h +0.0056, 6h +0.0055, 12h -0.0001, 24h +0.0019.
- O1 lr gap3 isotonic, seed 42: 1h +0.0049, 3h +0.0038, 6h +0.0045, 12h -0.0021, 24h +0.0008.
- O1 lr gap3 isotonic, seed 43: 1h +0.0048, 3h +0.0035, 6h +0.0043, 12h -0.0025, 24h +0.0004.
- O1 lr gap3 isotonic, seed 44: 1h +0.0046, 3h +0.0033, 6h +0.0044, 12h -0.0026, 24h +0.0005.
- O1 lr gap2 isotonic, seed 42: 1h -0.0013, 3h +0.0018, 6h +0.0028, 12h -0.0025, 24h -0.0047.
- O1 lr gap2 isotonic, seed 43: 1h -0.0005, 3h +0.0017, 6h +0.0030, 12h -0.0018, 24h -0.0039.
- O1 lr gap2 isotonic, seed 44: 1h -0.0004, 3h +0.0018, 6h +0.0028, 12h -0.0021, 24h -0.0041.
- O2 lr gap2 platt, seed 42: 1h +0.0057, 3h +0.0067, 6h +0.0057, 12h +0.0015, 24h +0.0028.
- O2 lr gap2 platt, seed 43: 1h +0.0053, 3h +0.0062, 6h +0.0054, 12h +0.0010, 24h +0.0023.
- O2 lr gap2 platt, seed 44: 1h +0.0057, 3h +0.0064, 6h +0.0056, 12h +0.0013, 24h +0.0027.
- O2 lr gap2 isotonic, seed 42: 1h +0.0044, 3h +0.0040, 6h +0.0031, 12h -0.0023, 24h +0.0001.
- O2 lr gap2 isotonic, seed 43: 1h +0.0042, 3h +0.0033, 6h +0.0026, 12h -0.0027, 24h -0.0003.
- O2 lr gap2 isotonic, seed 44: 1h +0.0046, 3h +0.0036, 6h +0.0029, 12h -0.0023, 24h +0.0001.
- O2 lr gap3 platt, seed 42: 1h +0.0029, 3h +0.0013, 6h +0.0024, 12h -0.0006, 24h +0.0002.
- O2 lr gap3 platt, seed 43: 1h +0.0027, 3h +0.0010, 6h +0.0025, 12h -0.0005, 24h +0.0002.
- O2 lr gap3 platt, seed 44: 1h +0.0030, 3h +0.0012, 6h +0.0026, 12h -0.0004, 24h +0.0001.
- O2 lr gap3 isotonic, seed 42: 1h +0.0058, 3h +0.0035, 6h +0.0025, 12h -0.0003, 24h -0.0036.
- O2 lr gap3 isotonic, seed 43: 1h +0.0059, 3h +0.0019, 6h +0.0024, 12h -0.0011, 24h -0.0035.
- O2 lr gap3 isotonic, seed 44: 1h +0.0061, 3h +0.0035, 6h +0.0027, 12h -0.0002, 24h -0.0033.

## Refit wall-clock cost — MEASURED (seeds 43/44 sweep runs)

Single in-situ measurement per pipeline fit, time.perf_counter around fit and transform separately (the cost_probe convention; same machine). The seed-42 sweep runs predate the instrumentation — the 6 h reference point is cost_probe/timings.json (median of 7 repeats on an origin-1 6 h slice). Medians [IQR] over both seeds, both origins and all models/buckets; apply is timed on the full test bucket.

| calib_hours | calibrator | n_fits | fit_time_s median [IQR] | apply_time_s median [IQR] | median n_calib |
|---|---|---|---|---|---|
| 1 | platt | 48 | 0.085 [0.067, 0.114] | 0.402 [0.357, 0.605] | 87,207 |
| 1 | isotonic | 48 | 0.032 [0.028, 0.039] | 0.403 [0.354, 0.514] | 87,207 |
| 3 | platt | 48 | 0.255 [0.201, 0.313] | 0.439 [0.346, 0.602] | 275,436 |
| 3 | isotonic | 48 | 0.105 [0.094, 0.125] | 0.420 [0.382, 0.561] | 275,436 |
| 12 | platt | 48 | 1.407 [1.094, 1.714] | 0.284 [0.153, 0.432] | 1,861,970 |
| 12 | isotonic | 48 | 0.872 [0.686, 1.014] | 0.420 [0.324, 0.586] | 1,861,970 |
| 24 | platt | 36 | 2.803 [2.350, 3.682] | 0.210 [0.133, 0.431] | 3,834,250 |
| 24 | isotonic | 36 | 2.112 [1.655, 2.741] | 0.473 [0.415, 0.587] | 3,834,250 |

## Verdict inputs — harm at 1 h vs 12 h (per seed x origin, pooled calibrators)

| seed | origin | harm@1h | harm@12h | mean dECE@1h | mean dECE@12h |
|---|---|---|---|---|---|
| 42 | 1 | 13/24 | 3/24 | +0.0018 | -0.0021 |
| 42 | 2 | 12/24 | 4/24 | -0.0002 | -0.0019 |
| 43 | 1 | 12/24 | 1/24 | +0.0007 | -0.0037 |
| 43 | 2 | 10/24 | 5/24 | -0.0002 | -0.0026 |
| 44 | 1 | 11/24 | 2/24 | +0.0005 | -0.0024 |
| 44 | 2 | 13/24 | 3/24 | +0.0012 | -0.0016 |
