# Tier-E Experiment A — mechanism control at seeds 43/44 (2026-10-23)

Exact replication of the Tier-D Exp 2 design (mech_slice.py) at training seeds 43 and 44: per bucket, Platt / isotonic fitted on (a) the full 12 h slice, (b) the true 1 h slice, (c) K=5 uniform subsamples of the 12 h slice at exactly the 1 h size (subsample RNG identical to seed 42: seeds 20261023 + origin*100000 + gap*1000 + k, shared across models and calibrators; the temporal prepare is seed-independent, so the drawn index sets are the same as seed 42's -- only the trained model differs). No per-impression predictions were saved for the seed 43/44 runs, so -- exactly as for seed 42 itself -- each (seed, origin) model set was retrained once with the run's own seeding replicated deterministically and validated against that seed's own sweep runs (see validation summary at the end). dECE = ECE(calibrated) - ECE(raw) on the bucket's fixed test set (the 12 h run's test split); harm = dECE > 0; the sub condition's per-cell dECE is the mean over its 5 subsamples. Recovery = (mean_c - mean_b) / (mean_a - mean_b): ~1 -> the size-matched 12 h-span subsample recovers the full slice's gain (span/coverage is the mechanism); ~0 -> it behaves like the 1 h slice (size is the mechanism).

## Seed 42

### All cells (gaps 0..+3; 48 cells per condition)

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | recovery |
|---|---|---|---|---|---|
| 1 | isotonic | -0.0031 | +0.0018 | -0.0023 | 0.84 |
| 1 | platt | -0.0010 | +0.0017 | -0.0008 | 0.93 |
| 2 | isotonic | -0.0019 | +0.0010 | -0.0016 | 0.89 |
| 2 | platt | -0.0019 | -0.0017 | -0.0018 | 0.68 |

Pooled over origins and calibrators: (a) -0.0020, (b) +0.0007, (c) -0.0016; recovery fraction 0.88. Harm counts of 48: full12 7, hour1 24, sub 7.

Per model (both origins pooled), all cells:

| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery |
|---|---|---|---|---|---|
| dcnv2 | isotonic | -0.0026 | +0.0016 | -0.0018 | 0.81 |
| dcnv2 | platt | -0.0015 | -0.0004 | -0.0014 | 0.88 |
| deepfm | isotonic | -0.0032 | +0.0001 | -0.0028 | 0.87 |
| deepfm | platt | -0.0020 | -0.0016 | -0.0019 | 0.74 |
| lr | isotonic | -0.0017 | +0.0025 | -0.0013 | 0.90 |
| lr | platt | -0.0008 | +0.0020 | -0.0007 | 0.96 |

### Fixed evaluation set (gaps +1..+3; 36 cells per condition) -- v8's paragraph set

Mean dECE: full12 -0.0024, hour1 -0.0001, sub -0.0021. Pooled recovery 0.87; recovery across origin x calibrator: o1 isotonic 0.80, o1 platt 0.83, o2 isotonic 0.93, o2 platt 0.99 (range 0.80-0.99). Harm counts of 36: full12 4, hour1 15, sub 4.

Per origin x calibrator (fixed set, 9 cells each):

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery | harm a/b/c |
|---|---|---|---|---|---|---|
| 1 | isotonic | -0.0032 | +0.0002 | -0.0025 | 0.80 | 1/4/1 |
| 1 | platt | -0.0010 | +0.0002 | -0.0008 | 0.83 | 2/3/2 |
| 2 | isotonic | -0.0033 | +0.0008 | -0.0030 | 0.93 | 0/6/0 |
| 2 | platt | -0.0021 | -0.0016 | -0.0021 | 0.99 | 1/2/1 |
## Seed 43

### All cells (gaps 0..+3; 48 cells per condition)

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | recovery |
|---|---|---|---|---|---|
| 1 | isotonic | -0.0044 | +0.0008 | -0.0034 | 0.80 |
| 1 | platt | -0.0029 | +0.0010 | -0.0027 | 0.96 |
| 2 | isotonic | -0.0032 | +0.0011 | -0.0029 | 0.92 |
| 2 | platt | -0.0020 | -0.0012 | -0.0019 | 0.93 |

Pooled over origins and calibrators: (a) -0.0031, (b) +0.0004, (c) -0.0027; recovery fraction 0.89. Harm counts of 48: full12 6, hour1 22, sub 8.

Per model (both origins pooled), all cells:

| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery |
|---|---|---|---|---|---|
| dcnv2 | isotonic | -0.0059 | -0.0013 | -0.0052 | 0.84 |
| dcnv2 | platt | -0.0049 | -0.0023 | -0.0048 | 0.94 |
| deepfm | isotonic | -0.0039 | +0.0015 | -0.0030 | 0.83 |
| deepfm | platt | -0.0018 | -0.0002 | -0.0017 | 0.94 |
| lr | isotonic | -0.0017 | +0.0026 | -0.0012 | 0.90 |
| lr | platt | -0.0007 | +0.0022 | -0.0005 | 0.96 |

### Fixed evaluation set (gaps +1..+3; 36 cells per condition) -- v8's paragraph set

Mean dECE: full12 -0.0028, hour1 +0.0001, sub -0.0025. Pooled recovery 0.88; recovery across origin x calibrator: o1 isotonic 0.74, o1 platt 0.93, o2 isotonic 0.93, o2 platt 0.96 (range 0.74-0.96). Harm counts of 36: full12 5, hour1 15, sub 7.

Per origin x calibrator (fixed set, 9 cells each):

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery | harm a/b/c |
|---|---|---|---|---|---|---|
| 1 | isotonic | -0.0043 | -0.0009 | -0.0034 | 0.74 | 0/3/0 |
| 1 | platt | -0.0025 | -0.0006 | -0.0023 | 0.93 | 1/3/2 |
| 2 | isotonic | -0.0032 | +0.0022 | -0.0029 | 0.93 | 1/5/2 |
| 2 | platt | -0.0012 | -0.0001 | -0.0012 | 0.96 | 3/4/3 |
## Seed 44

### All cells (gaps 0..+3; 48 cells per condition)

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub (mean of 5) | recovery |
|---|---|---|---|---|---|
| 1 | isotonic | -0.0032 | +0.0007 | -0.0022 | 0.76 |
| 1 | platt | -0.0015 | +0.0004 | -0.0013 | 0.87 |
| 2 | isotonic | -0.0017 | +0.0021 | -0.0015 | 0.94 |
| 2 | platt | -0.0014 | +0.0007 | -0.0013 | 0.97 |

Pooled over origins and calibrators: (a) -0.0020, (b) +0.0010, (c) -0.0016; recovery fraction 0.88. Harm counts of 48: full12 5, hour1 27, sub 7.

Per model (both origins pooled), all cells:

| model | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery |
|---|---|---|---|---|---|
| dcnv2 | isotonic | -0.0040 | +0.0003 | -0.0033 | 0.84 |
| dcnv2 | platt | -0.0031 | -0.0011 | -0.0029 | 0.87 |
| deepfm | isotonic | -0.0017 | +0.0014 | -0.0011 | 0.80 |
| deepfm | platt | -0.0006 | +0.0005 | -0.0005 | 0.91 |
| lr | isotonic | -0.0017 | +0.0026 | -0.0012 | 0.90 |
| lr | platt | -0.0007 | +0.0022 | -0.0006 | 0.97 |

### Fixed evaluation set (gaps +1..+3; 36 cells per condition) -- v8's paragraph set

Mean dECE: full12 -0.0022, hour1 +0.0007, sub -0.0019. Pooled recovery 0.88; recovery across origin x calibrator: o1 isotonic 0.72, o1 platt 0.75, o2 isotonic 0.95, o2 platt 1.00 (range 0.72-1.00). Harm counts of 36: full12 3, hour1 18, sub 4.

Per origin x calibrator (fixed set, 9 cells each):

| origin | calibrator | (a) full12 | (b) hour1 | (c) sub | recovery | harm a/b/c |
|---|---|---|---|---|---|---|
| 1 | isotonic | -0.0038 | -0.0007 | -0.0030 | 0.72 | 0/3/0 |
| 1 | platt | -0.0018 | -0.0007 | -0.0015 | 0.75 | 1/2/2 |
| 2 | isotonic | -0.0024 | +0.0024 | -0.0022 | 0.95 | 0/7/0 |
| 2 | platt | -0.0009 | +0.0017 | -0.0009 | 1.00 | 2/6/2 |
## Self-check (seed 42 recomputed in this code path vs published)

- all pooled a: recomputed -0.0020 vs published -0.0020 (OK)

- all pooled b: recomputed 0.0007 vs published 0.0007 (OK)

- all pooled c: recomputed -0.0016 vs published -0.0016 (OK)

- all recovery: recomputed 0.8768 vs published 0.8800 (OK)

- fixed a: recomputed -0.0024 vs published -0.0024 (OK)

- fixed b: recomputed -0.0001 vs published -0.0001 (OK)

- fixed c: recomputed -0.0021 vs published -0.0021 (OK)

- fixed recovery: recomputed 0.8745 vs published 0.8800 (OK)

- fixed rec range lo: recomputed 0.8026 vs published 0.8000 (OK)

- fixed rec range hi: recomputed 0.9935 vs published 0.9900 (OK)

- fixed harm full12/hour1/sub: recomputed 4/15/4 vs published 4/15/4 (OK)

## Cross-seed verdict

| seed | set | (a) full12 | (b) hour1 | (c) sub | recovery | rec range (o x cal) | harm a/b/c |
|---|---|---|---|---|---|---|---|
| 42 | all (48) | -0.0020 | +0.0007 | -0.0016 | 0.88 | -- | 7/24/7 |
| 42 | fixed (36) | -0.0024 | -0.0001 | -0.0021 | 0.87 | 0.80-0.99 | 4/15/4 |
| 43 | all (48) | -0.0031 | +0.0004 | -0.0027 | 0.89 | -- | 6/22/8 |
| 43 | fixed (36) | -0.0028 | +0.0001 | -0.0025 | 0.88 | 0.74-0.96 | 5/15/7 |
| 44 | all (48) | -0.0020 | +0.0010 | -0.0016 | 0.88 | -- | 5/27/7 |
| 44 | fixed (36) | -0.0022 | +0.0007 | -0.0019 | 0.88 | 0.72-1.00 | 3/18/4 |

Seed 42's coverage verdict (recovery 0.88 on both cell sets) HOLDS at seeds 43/44: fixed-set recovery 0.88 (seed 43) and 0.88 (seed 44); all-cells recovery 0.89 and 0.88. A recovery near 1 means the size-matched 12 h-span subsample behaves like the full 12 h slice, i.e. temporal span -- not impression count -- is the operative variable.

## Validation (retrain-once harness vs each seed's own sweep runs)

- seed 43 origin 1: full12 vs sweep-12h max |ddECE| = 9.541e-17; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 5.798e-13; 1h subset of 12h slice verified elementwise for all gaps.
- seed 43 origin 2: full12 vs sweep-12h max |ddECE| = 8.500e-17; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 9.436e-13; 1h subset of 12h slice verified elementwise for all gaps.
- seed 44 origin 1: full12 vs sweep-12h max |ddECE| = 1.261e-08; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 1.902e-13; 1h subset of 12h slice verified elementwise for all gaps.
- seed 44 origin 2: full12 vs sweep-12h max |ddECE| = 7.286e-17; hour1 vs sweep-1h (gaps 1-3) max |ddECE| = 2.329e-13; 1h subset of 12h slice verified elementwise for all gaps.
## Independent verification (2026-10-23, post-chain)

Recomputed from the per-cell CSVs (results/mechanism_slice/, results/mechanism_seeds/seed{43,44}/) with a separate script (sub = mean over k within cell first; recovery = (mean_sub - mean_hour1)/(mean_full12 - mean_hour1)). Every headline number in this file reproduces: seed 42 all-cells recovery 0.8768, fixed 0.8745, fixed means -0.0024/-0.0001/-0.0021, fixed harms 4/15/4 of 36; seed 43 all-cells 0.8854, fixed 0.8791, fixed harms 5/15/7 of 36, fixed o x cal range 0.744-0.957; seed 44 all-cells 0.8753, fixed 0.8794, fixed harms 3/18/4 of 36, fixed o x cal range 0.716-0.998. Seed 42 values also match the previously audited Tier-D numbers exactly. VERDICT: recovery holds at seeds 43/44 on both calibers (fixed-set 0.8745 / 0.8791 / 0.8794 for seeds 42/43/44) -- the size-matched coverage result is not a seed-42 accident; the paper's "seed-42 control" qualifier can be removed.
