# Calibration under temporal splits for CTR prediction (Avazu)

Experiment scaffold for the paper: **how CTR model calibration degrades as
the test period moves away from the training period under temporal splits,
whether cheap recalibration fixes it, and what the miscalibration costs in
a simulated second-price auction.**

## Research questions

1. **Replication** — how much do random splits inflate AUC / log loss compared
   with temporal splits? (Random-split arm vs the temporal gap-0 bucket.)
2. **Main** — how does calibration error (ECE, 15 bins) drift as the test
   period moves farther from the training period under temporal splits?
3. **Recalibration** — does recalibration (Platt scaling / isotonic regression,
   fitted per bucket on a small recent validation slice) recover calibration
   at each temporal distance?
4. **Bidding payoff** — in a simulated second-price auction (bid = predicted
   probability × value-per-click), what is the realized profit difference
   between raw vs recalibrated probabilities (vs an oracle upper bound)?

## Design

- **Data**: Avazu CTR logs (~10 days, 40.4M rows, `click` label, explicit
  `hour` timestamp `YYMMDDHH`). All non-time columns are treated as
  categorical, plus an engineered `hour_of_day` field.
- **Encoding**: per-field frequency-capped vocabulary (`min_count`,
  `max_vocab_per_field` in `config.yaml`), fit on each arm's **capped
  training rows only** (see the Plan-A pipeline below — this changed from
  the original scaffold, which fit on the full loaded training frame);
  unseen values map to UNK. The encoded global-index matrix is shared by
  all models (LR one-hots, deep-model embeddings).
- **Temporal protocol**: train on days 0–4 (capped, see below); test buckets
  are the **full** single days at gap 0 / +1 / +2 / +3 after the training
  window. Each bucket's recalibration slice is leak-free and immediately
  precedes it: for gap 0 it is the first `calib_hours` of the bucket day
  (test = the rest of that day); for gap ≥ 1 it is the last `calib_hours`
  of the previous day.
- **Models**: Logistic Regression (sklearn SGD, one-hot), DeepFM, DCN-v2
  (compact PyTorch implementations in `src/models.py`).
- **Metrics**: AUC, log loss, ECE (15 equal-width bins) + per-bin reliability
  tables exported for diagrams, computed for raw / Platt / isotonic variants.
- **Bidding simulation**: market (competing) price per impression drawn once
  from a lognormal (config) and shared across strategies, so comparisons are
  paired. Win if bid > market price, pay the market price, earn
  `value_per_click` on a realized click. The *oracle* strategy bids with
  ground-truth labels and is an unattainable upper bound only.

## Plan-A streaming pipeline (memory design)

The full Avazu CSV (6.3 GB, 40.4M rows) does not fit in this VM's RAM as a
DataFrame (7.9 GB total, no swap; budget: peak RSS < ~6.5 GB). The pipeline
in `src/data_pipeline.py` therefore never holds more than one CSV chunk
(`data.chunk_rows`, default 500k rows) of raw data:

1. **Pass 0 — dataset index**: one scan of the `hour` column only → total
   row count and per-(day, hour) counts, cached in
   `data/processed/dataset_index.json` (keyed on the raw file size). This
   makes every split size known exactly before any array is allocated.
2. **Per arm, scan 1 — vocab counts**: stream the CSV and accumulate exact
   per-field value counts over the arm's capped training rows only; fit the
   `VocabEncoder` from the counts (ties broken deterministically by count,
   then value string).
3. **Per arm, scan 2 — encode**: stream again, encode every needed row, and
   write into preallocated `.npy` memmaps under `data/processed/<arm>/`
   (features int32, labels int8). Training and evaluation stream batches
   from these memmaps with numpy mmap — the encoded full file (~3.4 GB on
   disk) is never materialised in RAM.

Training-row caps (`data.train_cap_rows`, default 4,000,000; CLI
`--train-cap` overrides):

- **Temporal arm**: deterministic stride sample of the days 0–4 window
  (stride = ⌈window rows / cap⌉, so the sample is ≤ cap rows). Test buckets
  and calibration slices are **full-size**. Note: because `min_count`
  applies to the sampled rows' counts, the effective frequency threshold
  scales with the stride (at cap 500k / stride 41, a value needs ~410
  window occurrences to reach a sampled count of 10; at cap 4M / stride 6,
  ~60). Vocabularies therefore grow with the cap (total_vocab ≈ 9k at
  500k cap; much larger at 4M).
- **Random arm**: a uniform random subset ("pool") of the file drawn by
  random-priority reservoir sampling (i.i.d. 64-bit priority per row, the
  smallest `pool_size` keys kept), sized so the 70% train share ≈ cap;
  the pool is then split 70/10/20 by a seeded permutation. The random
  arm's val/test sets are sample-based — only the temporal test buckets
  are full-size.

Prepared splits carry a **config fingerprint** (cap, seed, split and
vocab settings, row count) in `<arm>/manifest.json`. A rerun reuses the
prepare when the fingerprint matches and re-prepares that arm otherwise
(`--force-prepare` overrides). Memory, measured on the 500k-cap validation
run (full pipeline, cold): **peak RSS 2.60 GB**, of which the train+eval
phase alone peaks at 1.73 GB.

## LR solver

Logistic regression is trained with sklearn `SGDClassifier`
(`loss="log_loss"`) via `partial_fit` over streamed batches — the original
lbfgs default needed the full design in memory and did not converge within
`max_iter=300` even on 200k rows in the smoke run. SGD training
demonstrably finishes (fixed epochs over the capped train set) with
O(batch) memory. `alpha` mirrors the old `C` semantics:
`alpha = 1 / (C * n_train)`. The schedule is `adaptive` with `eta0=0.01`
and `average=True`: the `"optimal"` heuristic mis-scales at this alpha and
collapses the predicted probabilities toward 0 (measured at 500k train
rows, temporal gap 0: optimal → AUC 0.724 / logloss 1.96 / ECE 0.170;
adaptive eta0=0.01 → AUC 0.724 / logloss 0.423 / ECE 0.010). Deep models
train for a fixed number of epochs (no early stopping) so the temporal
comparison is not confounded by split-dependent stopping, with batches
streamed from the memmaps in a seeded per-epoch permutation order; torch
is seeded from the config seed and capped at `train.torch_threads`.

## Directory layout

```
avazu-temporal-calibration/
├── README.md
├── requirements.txt
├── config.yaml            # all experiment knobs (splits, caps, models, bidding)
├── run_all.sh             # data check + run entry point (uses .venv if present)
├── data/raw/              # put train.csv here (see Dataset below)
├── data/processed/       # prepared splits: dataset_index.json,
│                          # <arm>/{manifest.json, encoder.json, *_X.npy, *_y.npy}
├── results/               # metrics.csv, bidding.csv, per-arm CSVs,
│                          # reliability_*.csv, plots/, summary.md
└── src/
    ├── __init__.py
    ├── data_prep.py       # time parsing, VocabEncoder, legacy in-memory splits
    ├── data_pipeline.py   # Plan-A streaming prepare (pass 0 + 2 scans per arm)
    ├── models.py          # LR (SGD), DeepFM, DCN-v2 — streaming fit/predict
    ├── train_eval.py      # training helper + AUC/log loss/ECE evaluation
    ├── calibration.py     # ECE, reliability tables, Platt / isotonic
    ├── bidding_sim.py     # second-price payoff simulation
    └── run_experiments.py # orchestration: prepare -> per-arm train/eval -> merge
```

## Dataset

Avazu is distributed via Kaggle and requires a Kaggle account + API token
(`~/.kaggle/kaggle.json`). This scaffold does **not** download it for you.

With the Kaggle CLI:

```bash
pip install kaggle
mkdir -p data/raw
kaggle competitions download -c avazu-ctr-prediction -p data/raw
# the download is gzipped; produce data/raw/train.csv, e.g.:
gunzip -c data/raw/train.gz > data/raw/train.csv   # adjust name as downloaded
```

Manual fallback: download `train.gz` from the Kaggle competition page
("avazu-ctr-prediction") in a browser, unzip it, and place the file at
**`data/raw/train.csv`**. (The current `data/raw/train.csv` on this machine
came from a public mirror and is already in place.)

## Run order

```bash
pip install -r requirements.txt   # or use the existing .venv

# full run (config.yaml as-is: train cap 4M rows/arm, full-size temporal
# test buckets). THE full-run command:
./run_all.sh
# equivalent explicit form:
.venv/bin/python -m src.run_experiments --config config.yaml

# mid-size validation (what was run to validate Plan A; reuses prepared
# splits when the fingerprint matches, writes to a separate results dir):
.venv/bin/python -m src.run_experiments --config config.yaml \
    --train-cap 500000 --results-dir results_validation2

# smoke run (200k train rows/arm, 1 epoch, separate processed dir;
# temporal test buckets stay full-size, so this is a wiring check of
# several minutes, not seconds):
./run_all.sh --quick

# resume after an interruption: arms whose per-arm CSVs already exist in
# the results dir are skipped and their rows are loaded for the merge:
./run_all.sh --resume
```

Useful flags: `--train-cap N`, `--processed-dir PATH`, `--results-dir PATH`,
`--arm temporal|random`, `--resume`, `--force-prepare`.

## Outputs

| file | contents |
|---|---|
| `results/metrics_<arm>.csv`, `results/bidding_<arm>.csv` | per-arm results, written as each arm finishes |
| `results/metrics.csv` | AUC / log loss / ECE per (arm, model, bucket, variant), merged |
| `results/bidding.csv` | wins, value, cost, profit per (arm, model, bucket, strategy), merged |
| `results/reliability_*.csv` | per-bin reliability data (raw + recalibrated) |
| `results/plots/*.png` | reliability diagrams for the raw variant |
| `results/summary.md` | headline tables for RQ1, RQ2/RQ3, RQ4 |

## Validation (Plan A, 500k cap, 2026-10-20)

End-to-end run with `--train-cap 500000`, full-size temporal test buckets
(3.15M–5.29M rows per bucket), both arms, all 3 models: **wall 28.2 min
cold (incl. prepare), peak RSS 2.60 GB**. Directions vs the earlier
in-memory smoke run all reproduced: random-split AUC exceeds temporal
gap-0 AUC for every model (+0.014 to +0.016); raw ECE at gap +3 exceeds
gap 0 for every model (e.g. DeepFM 0.0076 → 0.0112, DCN-v2 0.0072 →
0.0114); recalibration at gap +3 lowers ECE in 5 of 6 model×method cases;
bidding profit per 1000 impressions is within ±0.2% between raw and
recalibrated strategies (oracle ≈ 2× at 131.0). Full numbers:
`results_validation2/` (an earlier same-day run with the broken LR
schedule is kept in `results_validation/` for the record). The run was
independently re-executed on 2026-10-20 into `results_validation3/`
(prepare reused via fingerprint): wall 20.3 min, train+eval peak RSS
1.64 GB, and agreement with `results_validation2/` within 1.4e-4 AUC /
1.1e-3 ECE on every row — the pipeline is reproducible at this cap.

## Notes / design decisions

- The encoder is fit per arm on that arm's capped training rows only.
- All randomness (reservoir keys, split permutation, batch order, market
  prices, torch init) is driven by `seed` in `config.yaml`; market prices
  are drawn once per bucket and shared across strategies.
- `data_prep.load_avazu` / `temporal_split` / `random_split` remain as the
  legacy in-memory reference implementation of the split protocol; the
  streaming pipeline implements the same protocol (same train window,
  buckets and calibration slices) without loading the file.
