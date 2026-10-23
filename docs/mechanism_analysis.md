# Tier-D Experiment 2 — slice size vs diurnal coverage (2026-10-22)

Seed 42, models trained once per origin (see mech_slice.py for the design and the validation against the sweep runs). Conditions per bucket: (a) full12 = fit on the full 12 h slice; (b) hour1 = fit on the actual 1 h slice (size n1); (c) sub = K=5 uniform subsamples of the 12 h slice at exactly size n1 (subsample seeds 20261023 + origin*100000 + gap*1000 + k, shared across models and calibrators). dECE = ECE(calibrated) - ECE(raw) on the bucket's fixed test set (the 12 h run's test split). Means are over the 12 model x gap cells per origin (sub: averaged over k within cell first). Recovery fraction = (mean_c - mean_b) / (mean_a - mean_b): ~1 -> the size-matched 12 h-coverage subsample recovers the full slice's gain, i.e. coverage (temporal span), not impression count, is the mechanism; ~0 -> it behaves like the 1 h slice, i.e. size is the mechanism.

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | (c) sd across k | recovery fraction |
|---|---|---|---|---|---|---|
| 1 | platt | -0.0010 | +0.0017 | -0.0008 | 0.0003 | 0.93 |
| 1 | isotonic | -0.0031 | +0.0018 | -0.0023 | 0.0008 | 0.84 |
| 2 | platt | -0.0019 | -0.0017 | -0.0018 | 0.0007 | 0.68 |
| 2 | isotonic | -0.0019 | +0.0010 | -0.0016 | 0.0008 | 0.89 |

Pooled over origins and calibrators: (a) -0.0020, (b) +0.0007, (c) -0.0016; recovery fraction 0.88.

## Per model (both origins pooled)

| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery fraction |
|---|---|---|---|---|---|
| lr | platt | -0.0008 | +0.0020 | -0.0007 | 0.96 |
| lr | isotonic | -0.0017 | +0.0025 | -0.0013 | 0.90 |
| deepfm | platt | -0.0020 | -0.0016 | -0.0019 | 0.74 |
| deepfm | isotonic | -0.0032 | +0.0001 | -0.0028 | 0.87 |
| dcnv2 | platt | -0.0015 | -0.0004 | -0.0014 | 0.88 |
| dcnv2 | isotonic | -0.0026 | +0.0016 | -0.0018 | 0.81 |

## Refit wall-clock cost — MEASURED (this experiment)

Single in-situ measurement per fit, time.perf_counter around fit and transform separately (the cost_probe convention; same machine). Apply is timed on the full test bucket. 'sub' pools the K=5 size-matched subsample fits (same n_fit as hour1).

| origin | calibrator | condition | n_fits | fit_time_s median [IQR] | apply_time_s median [IQR] |
|---|---|---|---|---|---|
| 1 | platt | full12 (12h) | 12 | 1.140 [1.049, 1.653] | 0.210 [0.122, 0.291] |
| 1 | platt | hour1 (1h) | 12 | 0.057 [0.046, 0.076] | 0.253 [0.156, 0.374] |
| 1 | platt | sub (1h-size, 12h span) | 60 | 0.052 [0.040, 0.073] | 0.200 [0.089, 0.250] |
| 1 | isotonic | full12 (12h) | 12 | 0.674 [0.558, 1.005] | 0.398 [0.326, 0.466] |
| 1 | isotonic | hour1 (1h) | 12 | 0.020 [0.019, 0.026] | 0.301 [0.240, 0.411] |
| 1 | isotonic | sub (1h-size, 12h span) | 60 | 0.021 [0.019, 0.027] | 0.321 [0.261, 0.426] |
| 2 | platt | full12 (12h) | 12 | 1.062 [0.970, 1.556] | 0.179 [0.100, 0.291] |
| 2 | platt | hour1 (1h) | 12 | 0.062 [0.054, 0.071] | 0.215 [0.136, 0.283] |
| 2 | platt | sub (1h-size, 12h span) | 60 | 0.056 [0.046, 0.076] | 0.174 [0.100, 0.223] |
| 2 | isotonic | full12 (12h) | 12 | 0.521 [0.500, 0.827] | 0.458 [0.375, 0.560] |
| 2 | isotonic | hour1 (1h) | 12 | 0.022 [0.019, 0.024] | 0.353 [0.272, 0.411] |
| 2 | isotonic | sub (1h-size, 12h span) | 60 | 0.023 [0.020, 0.026] | 0.361 [0.287, 0.411] |

Validation origin 1: full12 vs sweep-12h max |ddECE| = 8.07e-17; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 3.44e-13; 1h-subset-of-12h verified elementwise for all gaps (True); hour1 gap-0 vs sweep-1h differences (expected; different test remainder): {'lr/platt': 0.0006772420078571227, 'lr/isotonic': 0.0008981070077210863, 'deepfm/platt': -0.0005025685442214762, 'deepfm/isotonic': -0.00010117326211970829, 'dcnv2/platt': -0.0010826082378436498, 'dcnv2/isotonic': -0.0009188944160559761}.
Validation origin 2: full12 vs sweep-12h max |ddECE| = 8.59e-17; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 6.80e-13; 1h-subset-of-12h verified elementwise for all gaps (True); hour1 gap-0 vs sweep-1h differences (expected; different test remainder): {'lr/platt': 0.0003185669836057465, 'lr/isotonic': 0.0024878520128655207, 'deepfm/platt': -0.0034290518854360004, 'deepfm/isotonic': -0.002264236375154628, 'dcnv2/platt': 0.00032399428456491653, 'dcnv2/isotonic': 0.0002479830486226686}.
