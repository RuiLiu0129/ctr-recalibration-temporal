# Data

Nothing under `data/` is included in this repository.

- `data/raw/train.csv` — the Avazu training file (5.9 GB, 40,428,967 rows).
  It is distributed by Kaggle under the competition's terms and is **not
  redistributed here**. Download it from
  https://www.kaggle.com/competitions/avazu-ctr-prediction and place the
  decompressed file at `data/raw/train.csv`.
  Expected MD5: `f5d49ff28f41dc993b9ecb2372abb033`.
- `data/processed*/` — encoded splits (`.npy` memmaps) generated locally by
  `src/data_pipeline.py` from the raw CSV; they are build artifacts and are
  regenerated on each run (a config fingerprint decides reuse).
- Per-impression prediction arrays (`results/**/preds/*.npy`) are also
  excluded for size; all per-cell results derived from them are in
  `supplement/` as unrounded CSVs.
