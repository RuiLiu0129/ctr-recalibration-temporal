# Tier-E Experiment B — recency / staleness control (2026-10-23)

Seed 42, both origins, fixed buckets gaps +1..+3 (the 24 h sweep has no gap-0 bucket). Per bucket, Platt / isotonic are fitted on four disjoint 6-hour windows of the bucket's 24 h calibration slice (the whole day before the bucket), differing only in the data's age at serving time: age0 = slice-day hours 18-23 (the standard 6 h slice, [start-6h, start)), age6 = hours 12-17, age12 = hours 6-11, age18 = hours 0-5 ([start-24h, start-18h)). Span (6 h) and size class are held fixed; the full24 condition (the sweep's 24 h slice) is the reference. Models were trained once per origin (train split md5-identical across calib-hours settings). dECE = ECE(calibrated) - ECE(raw) on the full bucket day; harm = dECE > 0. Denominators: per origin x age = 18 cells (3 gaps x 3 models x 2 calibrators); pooled per age = 36 cells.

## Window sizes (impressions fitted on)

| origin | gap | age0 | age6 | age12 | age18 | full24 |
|---|---|---|---|---|---|---|
| 1 | 1 | 620,216 | 1,299,332 | 1,229,146 | 687,198 | 3,835,892 |
| 1 | 2 | 743,264 | 962,509 | 870,001 | 649,236 | 3,225,010 |
| 1 | 3 | 1,007,461 | 1,938,526 | 1,534,707 | 806,528 | 5,287,222 |
| 2 | 1 | 743,264 | 962,509 | 870,001 | 649,236 | 3,225,010 |
| 2 | 2 | 1,007,461 | 1,938,526 | 1,534,707 | 806,528 | 5,287,222 |
| 2 | 3 | 659,152 | 1,148,445 | 1,145,253 | 879,758 | 3,832,608 |

## Pooled (both origins; 36 cells per age, 36 for full24)

| condition | mean dECE | harm |
|---|---|---|
| age0 | -0.0003 | 13/36 |
| age6 | -0.0018 | 8/36 |
| age12 | +0.0004 | 19/36 |
| age18 | +0.0004 | 15/36 |
| full24 | -0.0019 | 9/36 |

## Origin 1 (18 cells per age)

| condition | mean dECE | harm |
|---|---|---|
| age0 | -0.0001 | 8/18 |
| age6 | -0.0014 | 3/18 |
| age12 | +0.0011 | 10/18 |
| age18 | +0.0008 | 10/18 |
| full24 | -0.0015 | 6/18 |

## Origin 2 (18 cells per age)

| condition | mean dECE | harm |
|---|---|---|
| age0 | -0.0005 | 5/18 |
| age6 | -0.0022 | 5/18 |
| age12 | -0.0004 | 9/18 |
| age18 | -0.0001 | 5/18 |
| full24 | -0.0023 | 3/18 |

## By calibrator (both origins pooled; 18 cells per age)

| calibrator | age0 | age6 | age12 | age18 | full24 |
|---|---|---|---|---|---|
| isotonic | -0.0004 | -0.0025 | -0.0004 | +0.0004 | -0.0029 |
| platt | -0.0003 | -0.0010 | +0.0011 | +0.0004 | -0.0009 |

## Paired per-cell contrast vs age0 (36 cells)

| contrast | mean d(dECE) | cells worse than age0 |
|---|---|---|
| age6 - age0 | -0.0014 | 13/36 |
| age12 - age0 | +0.0007 | 21/36 |
| age18 - age0 | +0.0007 | 23/36 |
| full24 - age0 | -0.0016 | 13/36 |

## Verdict

Pooled mean dECE by age: age0 -0.0003, age6 -0.0018, age12 +0.0004, age18 +0.0004; full24 reference -0.0019. The age sequence is NOT monotone pooled, and monotone in 0 of 2 origins. Age alone at fixed 6 h span does not reproduce the 24 h rebound's degradation, so staleness by itself is NOT a sufficient explanation of the rebound; the rebound must also involve the span/size change itself (e.g. hour-mix composition of the longer window).

## Refit wall-clock (median per fit, this experiment)

| condition | fit_time_s | apply_time_s |
|---|---|---|
| age0 | 0.292 | 0.420 |
| age6 | 0.576 | 0.408 |
| age12 | 0.434 | 0.319 |
| age18 | 0.251 | 0.342 |
| full24 | 1.867 | 0.316 |

## Validation

- origin 1: train md5 h24 == h12: {'X': True, 'y': True}; age0 window n == base 6 h slice n for all gaps: True; age0 vs base protocol max |ddECE| = 3.392e-14; full24 vs sweep-24h max |ddECE| = 8.587e-17; raw ECE vs base max |diff| = 9.801e-17.
- origin 2: train md5 h24 == h12: {'X': True, 'y': True}; age0 window n == base 6 h slice n for all gaps: True; age0 vs base protocol max |ddECE| = 1.649e-13; full24 vs sweep-24h max |ddECE| = 8.587e-17; raw ECE vs base max |diff| = 8.327e-17.
## Independent verification (2026-10-23, post-chain)

Recomputed from results/recency/recency_o{1,2}.csv with a separate script. Every figure in this file reproduces exactly: pooled mean dECE age0 -0.0003 (harm 13/36), age6 -0.0018 (8/36), age12 +0.0004 (19/36), age18 +0.0004 (15/36), full24 -0.0019 (9/36); origin 1: -0.0001 (8/18), -0.0014 (3/18), +0.0011 (10/18), +0.0008 (10/18); origin 2: -0.0005 (5/18), -0.0022 (5/18), -0.0004 (9/18), -0.0001 (5/18). VERDICT: the age trend at fixed 6 h span is NOT monotone (monotone in 0 of 2 origins) -- the 6 h-old window is best, the two oldest windows (12/18 h) turn net-harmful pooled, and the full-24 h slice (oldest average age) matches the best 6 h window. Staleness alone is therefore NOT a sufficient explanation of the 24 h rebound; threats' "untested" can be replaced by this measured result, with the design caveats that age is confounded with hour-of-day (age0 = 18-23h, age6 = 12-17h, age12 = 6-11h, age18 = 0-5h) and window impression counts differ by hour (age6 windows are the largest).
