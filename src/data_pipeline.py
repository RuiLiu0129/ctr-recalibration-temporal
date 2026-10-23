"""Plan-A streaming data pipeline: raw Avazu CSV -> encoded .npy splits on disk.

Motivation: the full Avazu file (40.4M rows, 6.3 GB) cannot be loaded as a
pandas DataFrame of strings on this machine (7.9 GB RAM, no swap). This
module never materialises more than one CSV chunk (``data.chunk_rows`` raw
rows) at a time:

  pass 0 (dataset index): one scan of the ``hour`` column only -> total row
      count + per-(day, hour) counts. Cached as ``dataset_index.json`` under
      the processed root (keyed on the raw file size).
  per arm:
    scan 1 (vocab counts): stream the CSV, accumulate exact per-field value
        counts over the arm's *capped* training rows only, and fit the
        frequency-capped ``VocabEncoder`` on those counts.
    scan 2 (encode): stream again, encode every needed row, and write it
        into preallocated ``.npy`` memmaps (features int32, labels int8)
        under ``<processed>/<arm>/``.

Splits (same protocol as ``data_prep.temporal_split`` / ``random_split``):
  temporal: train = deterministic stride sample (<= cap) of the
      ``train_days`` train window; test buckets = FULL days at gaps 0..3
      with the same leak-free calibration slices as the legacy design.
  random: a uniform random subset ("pool") of the file drawn by
      random-priority reservoir sampling (each row gets an i.i.d. 64-bit
      priority; the ``pool_size`` smallest keys are kept -- this is a
      standard reservoir equivalent), then a seeded 70/10/20 permutation
      split. The pool is sized so the train split holds ~``cap`` rows; the
      random arm's val/test sets are sample-based (only the temporal arm's
      test buckets are full-size).

Leak discipline is unchanged: each arm's encoder is fit on that arm's
(capped) training rows only, and a calibration slice strictly precedes its
test bucket, never overlapping the training window or its own test rows.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd

from .data_prep import CATEGORICAL_COLS, FEATURE_COLS, VocabEncoder

PIPELINE_VERSION = 2
USE_COLS = ["hour", "click"] + CATEGORICAL_COLS
_INT32_MAX_SAFE = 2**31 - 1


# ---------------------------------------------------------------------------
# pass 0: dataset index (hour column only)
# ---------------------------------------------------------------------------

def build_dataset_index(raw_path: str, processed_root: str, chunk_rows: int) -> dict:
    """Total row count + per-(day, hour) counts for the Avazu file.

    Uses the (month, day) code mmdd*100 + hour_of_day as a chunk-local key,
    then maps mmdd -> chronological day_idx after the scan. Reuses the
    cached index if the raw file size is unchanged.
    """
    raw_size = os.path.getsize(raw_path)
    cache_path = os.path.join(processed_root, "dataset_index.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("raw_size") == raw_size and cached.get("version") == PIPELINE_VERSION:
            print(f"dataset index: reusing cached {cache_path}")
            return cached

    counts: dict[int, int] = {}
    total = 0
    for chunk in pd.read_csv(raw_path, usecols=["hour"], chunksize=chunk_rows * 4):
        hour = chunk["hour"].to_numpy(dtype=np.int64)
        keys = ((hour // 100) % 10000) * 100 + (hour % 100)
        uniq, cnt = np.unique(keys, return_counts=True)
        for k, c in zip(uniq.tolist(), cnt.tolist()):
            counts[k] = counts.get(k, 0) + c
        total += len(hour)
    mmdds = sorted({k // 100 for k in counts})
    day_of = {m: i for i, m in enumerate(mmdds)}
    hour_counts = [[0] * 24 for _ in range(len(mmdds))]
    for k, c in counts.items():
        hour_counts[day_of[k // 100]][k % 100] = c
    index = {
        "version": PIPELINE_VERSION,
        "raw_size": raw_size,
        "total_rows": total,
        "mmdd": mmdds,
        "hour_counts": hour_counts,
    }
    os.makedirs(processed_root, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(index, f)
    print(f"dataset index: {total:,} rows over {len(mmdds)} days -> {cache_path}")
    return index


# ---------------------------------------------------------------------------
# per-chunk time helpers
# ---------------------------------------------------------------------------

def _day_lookup(index: dict) -> np.ndarray:
    """Map mmdd -> day_idx for mmdd in [0, 9999]."""
    lookup = np.full(10000, -1, dtype=np.int64)
    for i, m in enumerate(index["mmdd"]):
        lookup[m] = i
    return lookup


def _chunk_time(chunk: pd.DataFrame, day_lookup: np.ndarray):
    hour = chunk["hour"].to_numpy(dtype=np.int64)
    mmdd = ((hour // 100) % 10000).astype(np.int64)
    return day_lookup[mmdd], (hour % 100).astype(np.int64), chunk["click"].to_numpy(dtype=np.int64)


def _frame_for(chunk: pd.DataFrame, mask: np.ndarray, hod: np.ndarray) -> pd.DataFrame:
    """String-domain feature frame for the masked rows (pipeline encoding)."""
    sub = chunk.loc[mask, CATEGORICAL_COLS]
    return sub.assign(hour_of_day=hod[mask].astype(str))


def _chunk_iter(raw_path, chunk_rows, usecols=USE_COLS):
    dtypes = {"hour": np.int64, "click": np.int64}
    dtypes.update({c: str for c in CATEGORICAL_COLS})
    return pd.read_csv(
        raw_path, usecols=usecols, dtype=dtypes, na_filter=False,
        chunksize=chunk_rows,
    )


# ---------------------------------------------------------------------------
# reservoir (random-priority) pool selection
# ---------------------------------------------------------------------------

def _reservoir_pool(total: int, pool_size: int, seed: int, work_dir: str) -> np.ndarray:
    """Uniform random subset of {0..total-1} of size ``pool_size``.

    Each row gets an i.i.d. 64-bit priority; the ``pool_size`` smallest keys
    are kept. This is the classic random-priority reservoir sampler: the
    kept set is a uniform random subset, fully determined by ``seed``.
    """
    if pool_size >= total:
        return np.arange(total, dtype=np.int64)
    keys_path = os.path.join(work_dir, "_reservoir_keys.npy")
    keys = np.lib.format.open_memmap(keys_path, mode="w+", dtype=np.int64, shape=(total,))
    rng = np.random.default_rng(seed)
    step = 4_000_000
    for start in range(0, total, step):
        end = min(total, start + step)
        keys[start:end] = rng.integers(0, 2**62, size=end - start, dtype=np.int64)
    keys.flush()
    pool = np.sort(np.asarray(np.argpartition(keys, pool_size)[:pool_size], dtype=np.int64))
    del keys
    os.remove(keys_path)
    return pool


# ---------------------------------------------------------------------------
# fingerprinting
# ---------------------------------------------------------------------------

def _fingerprint(arm: str, cap: int | None, cfg: dict, total_rows: int) -> dict:
    split, feat = cfg["split"], cfg["features"]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "arm": arm,
        "cap": cap,
        "seed": int(cfg["seed"]),
        "total_rows": total_rows,
        "train_days": int(split["train_days"]),
        "train_start_day": int(split.get("train_start_day", 0)),
        "gaps": [int(g) for g in split["test_gaps_days"]],
        "calib_hours": int(split["calib_hours"]),
        "random_split": {k: float(v) for k, v in split["random"].items()},
        "min_count": int(feat["min_count"]),
        "max_vocab_per_field": int(feat["max_vocab_per_field"]),
    }


# ---------------------------------------------------------------------------
# split size math
# ---------------------------------------------------------------------------

def _temporal_sizes(index: dict, cfg: dict, cap: int | None):
    """Exact split row counts for the temporal arm."""
    hc = np.array(index["hour_counts"], dtype=np.int64)
    day_totals = hc.sum(axis=1)
    td = int(cfg["split"]["train_days"])
    start = int(cfg["split"].get("train_start_day", 0))
    ch = int(cfg["split"]["calib_hours"])
    gaps = [int(g) for g in cfg["split"]["test_gaps_days"]]
    window_rows = int(day_totals[start:start + td].sum())
    stride = max(1, math.ceil(window_rows / cap)) if cap else 1
    n_train = (window_rows + stride - 1) // stride
    sizes = {"train": n_train}
    for g in gaps:
        bd = start + td + g
        if g == 0:
            calib = int(hc[bd, :ch].sum())
            test = int(day_totals[bd]) - calib
        else:
            calib = int(hc[bd - 1, 24 - ch:].sum())
            test = int(day_totals[bd])
        sizes[f"calib_gap{g}"] = calib
        sizes[f"test_gap{g}"] = test
    return sizes, stride


# ---------------------------------------------------------------------------
# temporal arm prepare
# ---------------------------------------------------------------------------

def _prepare_temporal(arm_dir: str, raw_path: str, cfg: dict, index: dict,
                      cap: int | None, chunk_rows: int) -> dict:
    split_cfg = cfg["split"]
    td = int(split_cfg["train_days"])
    start = int(split_cfg.get("train_start_day", 0))
    ch = int(split_cfg["calib_hours"])
    gaps = [int(g) for g in split_cfg["test_gaps_days"]]
    day_lookup = _day_lookup(index)
    sizes, stride = _temporal_sizes(index, cfg, cap)
    print(f"temporal: stride={stride}, split sizes={{"
          + ", ".join(f"{k}: {v:,}" for k, v in sizes.items()) + "}")

    # -- scan 1: vocab counts over the stride-sampled train window ------------
    counters = [Counter() for _ in FEATURE_COLS]
    win_pos = 0
    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, _ = _chunk_time(chunk, day_lookup)
        in_window = (day >= start) & (day < start + td)
        wpos = np.arange(win_pos, win_pos + int(in_window.sum()))
        sel = np.zeros(len(chunk), dtype=bool)
        sel[np.flatnonzero(in_window)[wpos % stride == 0]] = True
        win_pos += int(in_window.sum())
        if sel.any():
            frame = _frame_for(chunk, sel, hod)
            for j, col in enumerate(FEATURE_COLS):
                counters[j].update(frame[col].value_counts().to_dict())
    encoder = VocabEncoder(
        int(cfg["features"]["max_vocab_per_field"]), int(cfg["features"]["min_count"])
    ).fit_from_counts(counters)
    del counters
    print(f"temporal: encoder total_vocab={encoder.total_vocab:,}")
    encoder.save(os.path.join(arm_dir, "encoder.json"))

    # -- scan 2: encode every needed row into preallocated memmaps -------------
    handles, offsets = {}, {}
    for name, n in sizes.items():
        handles[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_X.npy"), mode="w+",
            dtype=np.int32, shape=(n, len(FEATURE_COLS)))
        offsets[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_y.npy"), mode="w+",
            dtype=np.int8, shape=(n,))
    win_pos = 0
    offset = {k: 0 for k in sizes}

    def _put(name: str, mask: np.ndarray, chunk: pd.DataFrame, hod: np.ndarray, click: np.ndarray):
        if not mask.any():
            return
        X = encoder.transform_strings(_frame_for(chunk, mask, hod)).astype(np.int32)
        o = offset[name]
        handles[name][o:o + len(X)] = X
        offsets[name][o:o + len(X)] = click[mask].astype(np.int8)
        offset[name] = o + len(X)

    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, click = _chunk_time(chunk, day_lookup)
        in_window = (day >= start) & (day < start + td)
        wpos = np.arange(win_pos, win_pos + int(in_window.sum()))
        sel = np.zeros(len(chunk), dtype=bool)
        sel[np.flatnonzero(in_window)[wpos % stride == 0]] = True
        win_pos += int(in_window.sum())
        _put("train", sel, chunk, hod, click)
        for g in gaps:
            bd = start + td + g
            if g == 0:
                _put(f"calib_gap{g}", (day == bd) & (hod < ch), chunk, hod, click)
                _put(f"test_gap{g}", (day == bd) & (hod >= ch), chunk, hod, click)
            else:
                _put(f"calib_gap{g}", (day == bd - 1) & (hod >= 24 - ch), chunk, hod, click)
                _put(f"test_gap{g}", day == bd, chunk, hod, click)

    for name, h in handles.items():
        h.flush()
    for name, h in offsets.items():
        h.flush()
    for name, n in sizes.items():
        got = offset[name]
        if got != n:
            raise RuntimeError(
                f"temporal split {name}: expected {n:,} rows, wrote {got:,}")
    return {"splits": {name: {"n": n} for name, n in sizes.items()}}


# ---------------------------------------------------------------------------
# random arm prepare
# ---------------------------------------------------------------------------

def _prepare_random(arm_dir: str, raw_path: str, cfg: dict, index: dict,
                    cap: int | None, chunk_rows: int) -> dict:
    total = int(index["total_rows"])
    seed = int(cfg["seed"])
    rnd = cfg["split"]["random"]
    train_frac, val_frac = float(rnd["train_frac"]), float(rnd["val_frac"])
    pool_size = math.ceil(cap / train_frac) if cap else total
    pool = _reservoir_pool(total, int(pool_size), seed + 101, arm_dir)
    n_train = int(len(pool) * train_frac)
    n_val = int(len(pool) * val_frac)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pool))
    sets = {
        "train": np.sort(pool[perm[:n_train]]),
        "val": np.sort(pool[perm[n_train:n_train + n_val]]),
        "test": np.sort(pool[perm[n_train + n_val:]]),
    }
    del pool
    sizes = {name: len(s) for name, s in sets.items()}
    print(f"random: pool={len(sets['train']) + len(sets['val']) + len(sets['test']):,}, "
          "split sizes={" + ", ".join(f"{k}: {v:,}" for k, v in sizes.items()) + "}")
    day_lookup = _day_lookup(index)

    # -- scan 1: vocab counts over the train split rows -----------------------
    counters = [Counter() for _ in FEATURE_COLS]
    grow = 0
    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, _ = _chunk_time(chunk, day_lookup)
        gidx = np.arange(grow, grow + len(chunk))
        grow += len(chunk)
        sel = np.isin(gidx, sets["train"])
        if sel.any():
            frame = _frame_for(chunk, sel, hod)
            for j, col in enumerate(FEATURE_COLS):
                counters[j].update(frame[col].value_counts().to_dict())
    encoder = VocabEncoder(
        int(cfg["features"]["max_vocab_per_field"]), int(cfg["features"]["min_count"])
    ).fit_from_counts(counters)
    del counters
    print(f"random: encoder total_vocab={encoder.total_vocab:,}")
    encoder.save(os.path.join(arm_dir, "encoder.json"))

    # -- scan 2: encode --------------------------------------------------------
    handles, offsets = {}, {}
    for name, n in sizes.items():
        handles[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_X.npy"), mode="w+",
            dtype=np.int32, shape=(n, len(FEATURE_COLS)))
        offsets[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_y.npy"), mode="w+",
            dtype=np.int8, shape=(n,))
    grow = 0
    offset = {k: 0 for k in sizes}

    def _put(name: str, mask: np.ndarray, chunk: pd.DataFrame, hod: np.ndarray, click: np.ndarray):
        if not mask.any():
            return
        X = encoder.transform_strings(_frame_for(chunk, mask, hod)).astype(np.int32)
        o = offset[name]
        handles[name][o:o + len(X)] = X
        offsets[name][o:o + len(X)] = click[mask].astype(np.int8)
        offset[name] = o + len(X)

    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, click = _chunk_time(chunk, day_lookup)
        gidx = np.arange(grow, grow + len(chunk))
        grow += len(chunk)
        for name in ("train", "val", "test"):
            _put(name, np.isin(gidx, sets[name]), chunk, hod, click)

    for h in handles.values():
        h.flush()
    for h in offsets.values():
        h.flush()
    for name, n in sizes.items():
        if offset[name] != n:
            raise RuntimeError(
                f"random split {name}: expected {n:,} rows, wrote {offset[name]:,}")
    return {"splits": {name: {"n": n} for name, n in sizes.items()}}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def prepare_arm(arm: str, cfg: dict, processed_root: str, cap: int | None,
                force: bool = False) -> dict:
    """Prepare (or reuse) the encoded on-disk splits for one arm.

    Returns a manifest dict: fingerprint + split names/sizes + encoder info.
    Reuses an existing prepare when the stored fingerprint matches the
    requested one; rewrites the arm directory otherwise (or when force).
    """
    raw_path = cfg["data"]["raw_path"]
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Avazu file not found at {raw_path}")
    chunk_rows = int(cfg["data"].get("chunk_rows", 500_000))
    index = build_dataset_index(raw_path, processed_root, chunk_rows)

    arm_dir = os.path.join(processed_root, arm)
    manifest_path = os.path.join(arm_dir, "manifest.json")
    fp = _fingerprint(arm, cap, cfg, int(index["total_rows"]))
    if not force and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("fingerprint") == fp:
            print(f"{arm}: reusing prepared splits (fingerprint match)")
            return manifest
        print(f"{arm}: fingerprint mismatch -- re-preparing")

    if os.path.exists(arm_dir):
        shutil.rmtree(arm_dir)
    os.makedirs(arm_dir, exist_ok=True)

    if arm == "temporal":
        prep = _prepare_temporal(arm_dir, raw_path, cfg, index, cap, chunk_rows)
    elif arm == "random":
        prep = _prepare_random(arm_dir, raw_path, cfg, index, cap, chunk_rows)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    enc = VocabEncoder.load(os.path.join(arm_dir, "encoder.json"))
    manifest = {
        "fingerprint": fp,
        "encoder_path": os.path.join(arm_dir, "encoder.json"),
        "field_dims": enc.field_dims,
        "total_vocab": enc.total_vocab,
        **prep,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_split(arm_dir: str, name: str):
    """Load one prepared split as read-only memmaps (X int32, y int8)."""
    X = np.load(os.path.join(arm_dir, f"{name}_X.npy"), mmap_mode="r")
    y = np.load(os.path.join(arm_dir, f"{name}_y.npy"), mmap_mode="r")
    return X, y
