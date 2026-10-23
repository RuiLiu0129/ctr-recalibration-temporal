# How Much Fresh Data Should CTR Recalibration Use?

**A Temporal Evaluation of Post-hoc Calibration**

Rui Liu — Independent Researcher

Code and per-cell results for the paper *"How Much Fresh Data Should CTR
Recalibration Use? A Temporal Evaluation of Post-hoc Calibration"*, a study of
post-hoc calibration (Platt scaling, isotonic regression) for click-through
rate (CTR) prediction on the Avazu dataset, evaluated the way production
systems experience it: trained on the past, scored on future days, one day at
a time.

## What this repository contains

- `src/` — the experiment pipeline (streaming data preparation, LR / DeepFM /
  DCN-v2 models, calibration, ECE evaluation, and a simulated second-price
  auction). See `docs/PROJECT_README.md` for the full design notes, including
  the memory-bounded streaming pipeline and the solver choices.
- `config.yaml` and `config_*.yaml` — every experiment configuration used:
  the base protocol (6-hour recalibration slice), the second origin, the
  seed replicates (43, 44), and the slice-length sweep (1 / 3 / 6 / 12 / 24
  hours, both origins, all seeds).
- Analysis scripts at the top level:
  - `tierb_analyze.py` — the 48-cell result grid, slice/target bias
    statistics, and gate evaluations.
  - `tierc_analyze.py`, `tierc_finalize.py`, `tierc_ws5.py` — bootstrap
    confidence intervals for ΔECE, seed-stability analyses, slice-gate
    tests, and the slice-length sweep consolidation.
  - `mech_slice.py`, `stage_tierd.py`, `analyze_tierd.py` — the mechanism
    experiment (slice size vs. diurnal coverage) and multi-seed sweep
    staging/analysis.
  - `cost_probe/time_costs.py` — per-cycle recalibration timing probe.
- `supplement/` — the audit supplement: unrounded per-cell metrics for all
  seeds, slice and target signed biases, gate inputs and decisions, the
  slice-length sweep table, bootstrap intervals, and both test-day oracle
  tables, each indexed by `supplement/README.md` against the paper claim it
  supports. `supplement/reproduce_sign_corr.py` reproduces the paper's
  sign-agreement count (40 of 48) and the ΔECE–Δprofit correlations
  (pooled −0.69; origin 1 −0.77; origin 2 −0.78) from the supplement CSVs.
  The Tier-D tables (three-seed slice-length sweep, mechanism control) are
  listed in the Tier-D section of `supplement/README.md`.
- `requirements.txt` — the pinned environment (see below).
- `docs/` — the original project README (`PROJECT_README.md`, design and
  protocol details) plus the Tier-D analysis write-ups:
  `sweep_seeds_analysis.md` (slice-length sweep at seeds 43/44) and
  `mechanism_analysis.md` (slice size vs. diurnal coverage, with measured
  refit costs).

## Dataset

The experiments use the Avazu CTR dataset (40,428,967 impressions,
Oct 21–30, 2014), distributed by Kaggle as the
[avazu-ctr-prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction)
competition. **The raw dataset is not redistributed here** (it is subject to
the competition's terms). To reproduce:

1. Download `train.gz` from the competition page (a Kaggle account is
   required) and decompress it to `data/raw/train.csv`.
2. Verify the file: MD5 of `train.csv` is
   `f5d49ff28f41dc993b9ecb2372abb033`.

Per-impression prediction files produced by the runs are likewise not
included (they are large binary arrays); every per-cell number reported in
the paper is in `supplement/` in unrounded form.

## Environment

The pinned environment that produced all reported results
(`requirements.txt`):

- Python 3.12.3
- PyTorch 2.14.1+cpu
- scikit-learn 1.9.1
- NumPy 2.5.4
- SciPy 1.18.1
- pandas 3.0.6

Experiments ran on a 2-vCPU host (AMD EPYC 9D25); the pipeline is
memory-bounded by design and streams the 5.9 GB raw CSV in chunks.

## Reproducing

```bash
pip install -r requirements.txt

# Full base run (config.yaml: 4M-row training cap per arm, both arms,
# full-size temporal test buckets):
./run_all.sh
# equivalent explicit form:
python -m src.run_experiments --config config.yaml

# A variant, e.g. origin 2, seed 43, or one sweep point:
python -m src.run_experiments --config config_tierb_o2.yaml
python -m src.run_experiments --config config_sweep_o1_h12.yaml
```

Runs are resumable: arms whose per-arm CSVs already exist in the results
directory are skipped and merged from disk (`--resume`). The analysis
scripts and chains are driven by `run_tierb.sh`, `run_tierc.sh`, and
`run_tierd.sh`. Headline numbers can be checked directly against
`supplement/` without re-running anything.

## Protocol in brief

- Two rolling temporal origins on Avazu: origin 1 trains on days 0–4 and is
  evaluated on days 5–8 (gaps 0 to +3); origin 2 shifts the window by one
  day. A size-matched random-split arm serves as the conventional-protocol
  reference. Three training seeds (42, 43, 44) per arm.
- For each test day, Platt scaling and isotonic regression are fitted on
  the hours immediately preceding that day (base protocol: 6 hours; the
  sweep varies this from 1 to 24 hours) and applied leak-free.
- Calibration is measured with ECE over 15 equal-width bins; downstream
  impact is measured in a simulated second-price auction with a lognormal
  market price (μ = −2.2, σ = 0.9) and value per click 1.0.
