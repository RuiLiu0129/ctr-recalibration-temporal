"""Data loading, time parsing, categorical encoding, and split protocols for Avazu.

Avazu (Kaggle `avazu-ctr-prediction`) has 24 columns. The `hour` column is an
integer/string of the form YYMMDDHH (e.g. 14102100 = 2014-10-21 00:00) spanning
10 days, which is what makes the temporal-split protocol possible.

Split protocol (temporal arm)
-----------------------------
- Train: the first ``split.train_days`` days (day_idx 0 .. train_days-1).
- For each gap g in ``split.test_gaps_days`` the test bucket is the single day
  ``train_days + g``.
- Recalibration slice (``split.calib_hours`` hours, leak-free: never part of
  the training window and never part of the bucket it calibrates):
    * g == 0: the first ``calib_hours`` hours of the bucket day;
      the bucket's test rows are the remaining hours of that day.
    * g >= 1: the last ``calib_hours`` hours of the day before the bucket;
      the bucket's test rows are the full bucket day.

Random arm (replication): a seeded shuffle split; the validation slice doubles
as the recalibration slice.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# All Avazu feature columns except id / click / hour, plus the engineered
# hour_of_day. C1 and C14-C21 are treated as categorical, as is standard for
# this dataset.
CATEGORICAL_COLS = [
    "C1", "banner_pos",
    "site_id", "site_domain", "site_category",
    "app_id", "app_domain", "app_category",
    "device_id", "device_ip", "device_model", "device_type", "device_conn_type",
    "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
]
FEATURE_COLS = CATEGORICAL_COLS + ["hour_of_day"]
LABEL_COL = "click"


def _resolve_raw_path(raw_path: str) -> str:
    if os.path.exists(raw_path):
        return raw_path
    for candidate in (raw_path + ".gz", os.path.join(os.path.dirname(raw_path), "train.gz")):
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Avazu training file not found at {raw_path} (also tried .gz variants). "
        "See README.md, 'Dataset' section."
    )


def load_avazu(raw_path: str, nrows: int | None = None,
               sample_stride: int | None = None) -> pd.DataFrame:
    """Load the Avazu training CSV and add parsed time columns.

    Adds: ``day_idx`` (0-based, chronological), ``hour_of_day`` (0-23),
    ``abs_hour`` (hours since the first hour in the file). Rows are sorted by
    ``abs_hour``. Note: with ``nrows`` set, only the first ``nrows`` file rows
    are read, which for the (chronologically ordered) Avazu file means an
    early-days subset -- fine for smoke runs, not for real temporal results.

    With ``sample_stride`` > 1 the file is instead streamed in chunks and
    every ``sample_stride``-th row is kept, so the sample spans the file's
    full time range (used by the --quick smoke run); ``nrows`` then caps the
    number of *kept* rows.
    """
    path = _resolve_raw_path(raw_path)
    if sample_stride is not None and sample_stride > 1:
        parts, kept = [], 0
        for chunk in pd.read_csv(path, chunksize=1_000_000):
            part = chunk.iloc[::sample_stride]
            if nrows is not None:
                part = part.iloc[: nrows - kept]
            parts.append(part)
            kept += len(part)
            if nrows is not None and kept >= nrows:
                break
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(path, nrows=nrows)
    missing = [c for c in ("hour", LABEL_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected Avazu columns: {missing}")

    hour_str = df["hour"].astype(str).str.zfill(8)
    month = hour_str.str.slice(2, 4).astype(int)
    day = hour_str.str.slice(4, 6).astype(int)
    df["hour_of_day"] = hour_str.str.slice(6, 8).astype(int)

    # Map the (month, day) pairs present in the file to chronological 0..D-1.
    date_key = month * 100 + day
    ordered_dates = np.sort(date_key.unique())
    day_index_of = {int(d): i for i, d in enumerate(ordered_dates)}
    df["day_idx"] = date_key.map(day_index_of).astype(int)
    df["abs_hour"] = df["day_idx"] * 24 + df["hour_of_day"]

    df = df.sort_values("abs_hour", kind="stable").reset_index(drop=True)
    return df


class VocabEncoder:
    """Per-field frequency-capped categorical encoder.

    Each field gets a local vocabulary: 0 = UNK, 1..V-1 = the most frequent
    values (count >= min_count, at most max_vocab_per_field - 1 of them).
    ``transform`` returns *global* indices (local index + field offset), which
    is the index space shared by the embedding tables and the LR one-hot
    builder. Fit on a training split only, so test-only values map to UNK.
    """

    def __init__(self, max_vocab_per_field: int = 20000, min_count: int = 10):
        self.max_vocab_per_field = max_vocab_per_field
        self.min_count = min_count
        self.field_dims: list[int] = []
        self.field_offsets: list[int] = []
        self.total_vocab: int = 0
        self._maps: list[dict] = []

    def fit(self, df: pd.DataFrame) -> "VocabEncoder":
        self._maps, self.field_dims, self.field_offsets = [], [], []
        offset = 0
        for col in FEATURE_COLS:
            counts = df[col].value_counts()
            counts = counts[counts >= self.min_count]
            top = counts.index[: self.max_vocab_per_field - 1]
            self._maps.append({value: i + 1 for i, value in enumerate(top)})
            dim = len(top) + 1  # + UNK
            self.field_dims.append(dim)
            self.field_offsets.append(offset)
            offset += dim
        self.total_vocab = offset
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._maps:
            raise RuntimeError("VocabEncoder.transform called before fit")
        n = len(df)
        out = np.empty((n, len(FEATURE_COLS)), dtype=np.int64)
        for j, col in enumerate(FEATURE_COLS):
            mapping = self._maps[j]
            local = df[col].map(mapping).fillna(0).to_numpy(dtype=np.int64)
            out[:, j] = local + self.field_offsets[j]
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def fit_from_counts(self, counts_per_field: list[dict]) -> "VocabEncoder":
        """Fit from pre-accumulated per-field value counts (Plan-A pipeline).

        ``counts_per_field[j]`` maps raw value -> occurrence count for
        ``FEATURE_COLS[j]``, accumulated over the arm's (capped) training
        rows only. Values are ranked by count descending; ties break on the
        string form of the value so the vocabulary is deterministic across
        runs (the legacy ``fit`` inherits pandas ``value_counts`` tie order,
        so index assignments can differ from the legacy path -- vocabularies
        are equivalent, index numbering is not guaranteed identical).
        """
        if len(counts_per_field) != len(FEATURE_COLS):
            raise ValueError("counts_per_field must have one entry per feature column")
        self._maps, self.field_dims, self.field_offsets = [], [], []
        offset = 0
        for counts in counts_per_field:
            items = [(v, c) for v, c in counts.items() if c >= self.min_count]
            items.sort(key=lambda vc: (-vc[1], str(vc[0])))
            top = items[: self.max_vocab_per_field - 1]
            self._maps.append({value: i + 1 for i, (value, _) in enumerate(top)})
            dim = len(top) + 1  # + UNK
            self.field_dims.append(dim)
            self.field_offsets.append(offset)
            offset += dim
        self.total_vocab = offset
        return self

    def transform_strings(self, df: pd.DataFrame) -> np.ndarray:
        """``transform`` for frames whose feature values are strings.

        Used by the streaming pipeline (``data_pipeline``), where every
        feature column -- including the engineered ``hour_of_day`` -- is
        carried as a string. Implemented via per-column factorisation so the
        (potentially large) per-row work is a single array take; only the
        distinct values in the frame go through the Python dict.
        """
        if not self._maps:
            raise RuntimeError("VocabEncoder.transform_strings called before fit")
        n = len(df)
        out = np.empty((n, len(FEATURE_COLS)), dtype=np.int64)
        for j, col in enumerate(FEATURE_COLS):
            mapping = self._maps[j]
            codes, uniques = pd.factorize(df[col].astype(str))
            local_of_unique = np.fromiter(
                (mapping.get(u, 0) for u in uniques), dtype=np.int64, count=len(uniques)
            )
            local = np.where(codes >= 0, local_of_unique[np.maximum(codes, 0)], 0)
            out[:, j] = local + self.field_offsets[j]
        return out

    def save(self, path: str) -> None:
        """Persist the fitted vocabulary as JSON (values stored as strings)."""
        payload = {
            "max_vocab_per_field": self.max_vocab_per_field,
            "min_count": self.min_count,
            "field_dims": self.field_dims,
            "field_offsets": self.field_offsets,
            "total_vocab": self.total_vocab,
            "vocab": [
                [str(v) for v, _ in sorted(m.items(), key=lambda kv: kv[1])]
                for m in self._maps
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "VocabEncoder":
        with open(path) as f:
            payload = json.load(f)
        enc = cls(payload["max_vocab_per_field"], payload["min_count"])
        enc.field_dims = [int(d) for d in payload["field_dims"]]
        enc.field_offsets = [int(o) for o in payload["field_offsets"]]
        enc.total_vocab = int(payload["total_vocab"])
        enc._maps = [
            {value: i + 1 for i, value in enumerate(values)}
            for values in payload["vocab"]
        ]
        return enc


def temporal_split(df: pd.DataFrame, cfg: dict) -> dict:
    """Build the temporal arm.

    Returns a dict with:
      train  : DataFrame for day_idx < train_days
      buckets: {gap: {"test": DataFrame, "calib": DataFrame}}
    """
    split_cfg = cfg["split"]
    train_days = int(split_cfg["train_days"])
    calib_hours = int(split_cfg["calib_hours"])

    train = df[df["day_idx"] < train_days]
    buckets: dict[int, dict[str, pd.DataFrame]] = {}
    max_day = int(df["day_idx"].max())
    for gap in split_cfg["test_gaps_days"]:
        gap = int(gap)
        bucket_day = train_days + gap
        if bucket_day > max_day:
            raise ValueError(
                f"test bucket day {bucket_day} (gap {gap}) exceeds the last "
                f"day in the data (day_idx {max_day}); reduce test_gaps_days "
                "or provide more days of data"
            )
        if gap == 0:
            day_rows = df[df["day_idx"] == bucket_day]
            calib = day_rows[day_rows["hour_of_day"] < calib_hours]
            test = day_rows[day_rows["hour_of_day"] >= calib_hours]
        else:
            prev = df[df["day_idx"] == bucket_day - 1]
            calib = prev[prev["hour_of_day"] >= 24 - calib_hours]
            test = df[df["day_idx"] == bucket_day]
        buckets[gap] = {"test": test, "calib": calib}
    return {"train": train, "buckets": buckets}


def random_split(df: pd.DataFrame, cfg: dict) -> dict:
    """Build the random (replication) arm: seeded shuffle split.

    Returns {"train", "val", "test"} DataFrames. ``val`` is the recalibration
    slice for this arm.
    """
    rnd_cfg = cfg["split"]["random"]
    rng = np.random.default_rng(int(cfg["seed"]))
    idx = rng.permutation(len(df))
    n_train = int(len(df) * float(rnd_cfg["train_frac"]))
    n_val = int(len(df) * float(rnd_cfg["val_frac"]))
    return {
        "train": df.iloc[idx[:n_train]],
        "val": df.iloc[idx[n_train:n_train + n_val]],
        "test": df.iloc[idx[n_train + n_val:]],
    }


def encode_frames(encoder: VocabEncoder, frames: dict[str, pd.DataFrame]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Encode several DataFrames with a fitted encoder.

    Returns {name: (X, y)} with X the (n, n_fields) global-index matrix and y
    the float32 label vector.
    """
    out = {}
    for name, frame in frames.items():
        X = encoder.transform(frame)
        y = frame[LABEL_COL].to_numpy(dtype=np.float32)
        out[name] = (X, y)
    return out
