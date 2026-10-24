"""Tier-F Experiment B data preparation (2026-10-23): the random-draw arm
for the same-target-day identification control.

For one (origin, seed): draw uniformly at random WITHOUT replacement from
the origin's 5-day training window (the same history pool the temporal
arm's stride sample is drawn from), with the row count EXACTLY equal to
the temporal arm's training size for that origin (stride-cap math in
src/data_pipeline._temporal_sizes; asserted against the temporal arm's
own manifest). ALL preprocessing is fit inside the arm, mirroring
src.data_pipeline.prepare_arm: the VocabEncoder is fit on the drawn rows
only, and the four target test buckets are encoded with THIS arm's
encoder. Test buckets are row-identical to the temporal arm's (same
definitions: gap 0 = bucket day minus its first calib_hours, which are
the calibration slice there; gaps 1-3 = the full bucket day) -- asserted
against the temporal manifest's split sizes.

Draw rule (recorded in the manifest): 
    rng = np.random.default_rng(seed)
    chosen = sort(rng.choice(window_rows, size=n_train, replace=False))
i.e. the run's own seed, the same seed the models are trained with.

Usage:
    .venv/bin/python tierf_prep.py --origin 1 --seed 42
Output: data/processed_tierf/o{origin}_s{seed}/tierf/{encoder.json,
train_X/y.npy, test_gap{0..3}_X/y.npy, manifest.json}. Re-running with
an existing manifest is a no-op (fingerprint check).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import yaml

from src.data_pipeline import _chunk_iter, _chunk_time, _day_lookup, _frame_for
from src.data_prep import FEATURE_COLS, VocabEncoder

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPORAL_MANIFEST = {
    1: "data/processed_o1repro/temporal/manifest.json",
    2: "data/processed_o2/temporal/manifest.json",
}
INDEX_PATH = "data/processed/dataset_index.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", type=int, required=True, choices=(1, 2))
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    origin, seed = args.origin, args.seed
    start = origin - 1  # origin 1 -> day 0, origin 2 -> day 1 (as temporal)

    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    td = int(cfg["split"]["train_days"])
    ch = int(cfg["split"]["calib_hours"])
    gaps = [int(g) for g in cfg["split"]["test_gaps_days"]]
    cap = int(cfg["data"]["train_cap_rows"])
    chunk_rows = int(cfg["data"].get("chunk_rows", 500_000))
    raw_path = os.path.join(ROOT, cfg["data"]["raw_path"])

    out_root = os.path.join(
        ROOT, "data", "processed_tierf", f"o{origin}_s{seed}")
    arm_dir = os.path.join(out_root, "tierf")
    manifest_path = os.path.join(arm_dir, "manifest.json")

    with open(os.path.join(ROOT, INDEX_PATH)) as f:
        index = json.load(f)
    hc = np.array(index["hour_counts"], dtype=np.int64)
    day_totals = hc.sum(axis=1)
    window_rows = int(day_totals[start:start + td].sum())

    # temporal-arm training size, recomputed from the stride-cap math and
    # asserted against the temporal arm's own manifest
    stride = max(1, math.ceil(window_rows / cap))
    n_train = (window_rows + stride - 1) // stride
    with open(os.path.join(ROOT, TEMPORAL_MANIFEST[origin])) as f:
        tman = json.load(f)
    assert n_train == int(tman["splits"]["train"]["n"]), (
        f"n_train mismatch: computed {n_train:,} vs temporal manifest "
        f"{tman['splits']['train']['n']:,}")
    sizes = {"train": n_train}
    for g in gaps:
        bd = start + td + g
        if g == 0:
            sizes[f"test_gap{g}"] = int(day_totals[bd]) - int(hc[bd, :ch].sum())
        else:
            sizes[f"test_gap{g}"] = int(day_totals[bd])
        assert sizes[f"test_gap{g}"] == int(tman["splits"][f"test_gap{g}"]["n"]), (
            f"test_gap{g} size mismatch vs temporal manifest")

    fingerprint = {
        "arm": "tierf_same_day_random_draw", "origin": origin, "seed": seed,
        "train_start_day": start, "window_rows": window_rows,
        "n_train": n_train, "stride_cap_check": stride,
        "draw": "default_rng(seed).choice(window_rows, n_train, "
                "replace=False), sorted",
        "sizes": sizes,
    }
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old = json.load(f)
        if old.get("fingerprint") == fingerprint:
            print(f"tierf o{origin} s{seed}: reusing prepared splits "
                  f"(fingerprint match)", flush=True)
            return
        raise RuntimeError("existing tierf manifest with different "
                           "fingerprint -- refusing to overwrite")

    os.makedirs(arm_dir, exist_ok=True)
    day_lookup = _day_lookup(index)

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(window_rows, size=n_train, replace=False))
    print(f"tierf o{origin} s{seed}: window={window_rows:,} rows, "
          f"drew {n_train:,} (stride-cap equivalent {stride})", flush=True)

    # -- scan 1: vocab counts over the drawn rows only ---------------------
    counters = [Counter() for _ in FEATURE_COLS]
    win_pos = 0
    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, _ = _chunk_time(chunk, day_lookup)
        in_window = (day >= start) & (day < start + td)
        n_in = int(in_window.sum())
        if n_in:
            wpos = np.arange(win_pos, win_pos + n_in)
            keep = np.isin(wpos, chosen)
            if keep.any():
                sel = np.zeros(len(chunk), dtype=bool)
                sel[np.flatnonzero(in_window)[keep]] = True
                frame = _frame_for(chunk, sel, hod)
                for j, col in enumerate(FEATURE_COLS):
                    counters[j].update(frame[col].value_counts().to_dict())
        win_pos += n_in
    encoder = VocabEncoder(
        int(cfg["features"]["max_vocab_per_field"]),
        int(cfg["features"]["min_count"])).fit_from_counts(counters)
    del counters
    print(f"tierf: encoder total_vocab={encoder.total_vocab:,}", flush=True)
    encoder.save(os.path.join(arm_dir, "encoder.json"))

    # -- scan 2: encode drawn train rows + the four test buckets -----------
    handles, offsets, offset = {}, {}, {}
    for name, n in sizes.items():
        handles[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_X.npy"), mode="w+",
            dtype=np.int32, shape=(n, len(FEATURE_COLS)))
        offsets[name] = np.lib.format.open_memmap(
            os.path.join(arm_dir, f"{name}_y.npy"), mode="w+",
            dtype=np.int8, shape=(n,))
        offset[name] = 0

    def _put(name, mask, chunk, hod, click):
        if not mask.any():
            return
        X = encoder.transform_strings(
            _frame_for(chunk, mask, hod)).astype(np.int32)
        o = offset[name]
        handles[name][o:o + len(X)] = X
        offsets[name][o:o + len(X)] = click[mask].astype(np.int8)
        offset[name] = o + len(X)

    win_pos = 0
    for chunk in _chunk_iter(raw_path, chunk_rows):
        day, hod, click = _chunk_time(chunk, day_lookup)
        in_window = (day >= start) & (day < start + td)
        n_in = int(in_window.sum())
        if n_in:
            wpos = np.arange(win_pos, win_pos + n_in)
            keep = np.isin(wpos, chosen)
            if keep.any():
                sel = np.zeros(len(chunk), dtype=bool)
                sel[np.flatnonzero(in_window)[keep]] = True
                _put("train", sel, chunk, hod, click)
        win_pos += n_in
        for g in gaps:
            bd = start + td + g
            if g == 0:
                _put(f"test_gap{g}", (day == bd) & (hod >= ch),
                     chunk, hod, click)
            else:
                _put(f"test_gap{g}", day == bd, chunk, hod, click)

    for h in list(handles.values()) + list(offsets.values()):
        h.flush()
    for name, n in sizes.items():
        if offset[name] != n:
            raise RuntimeError(
                f"tierf split {name}: expected {n:,}, wrote {offset[name]:,}")
    manifest = {
        "fingerprint": fingerprint,
        "encoder_path": os.path.join(arm_dir, "encoder.json"),
        "field_dims": encoder.field_dims,
        "total_vocab": encoder.total_vocab,
        "splits": {name: {"n": n} for name, n in sizes.items()},
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"tierf o{origin} s{seed}: prepared. sizes="
          + ", ".join(f"{k}: {v:,}" for k, v in sizes.items()), flush=True)


if __name__ == "__main__":
    main()
