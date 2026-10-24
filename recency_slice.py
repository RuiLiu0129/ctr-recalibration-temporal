"""Tier-E Experiment B (2026-10-23): recency / staleness control.

Tests the staleness reading of the sweep's 24 h rebound. In the slice-
length sweep, lengthening the slice changes three things at once: size,
temporal span, and average AGE of the fitting data at serving time. The
mechanism control (Tier-D Exp 2 / Tier-E Exp A) separated span from
size; this experiment holds span (6 h) and size class (~0.6-1.0M
impressions) fixed and varies only age.

Design (seed 42, both origins, fixed buckets gaps +1..+3 -- the 24 h
sweep has no gap-0 bucket): train each model ONCE per origin on the
origin's train split (md5-verified identical across calib-hours
settings), score the bucket's full 24 h calibration slice (the sweep
h24 run's slice: the whole day before the bucket), and fit Platt /
isotonic on four disjoint 6-hour windows of that slice, by hour-of-day
of the slice day:

  age0   hod 18-23  = [start- 6h, start)     -- the standard 6 h slice
  age6   hod 12-17  = [start-12h, start- 6h)
  age12  hod  6-11  = [start-18h, start-12h)
  age18  hod  0- 5  = [start-24h, start-18h)

plus a full24 reference condition (the sweep's 24 h condition, all 24
hod blocks). Window membership is decoded from the encoded hour_of_day
feature via the arm's encoder.json (rows in a calib split are
chronological; hod is verified non-decreasing and each window's hod set
is asserted to be exactly its six hours).

All conditions are evaluated on the bucket's fixed test set (the h24
run's test split = the full bucket day, identical to every other
calib-hours setting for gaps >= 1).

Validation (written to recency_o{o}_validation.json):
  * age0's window is row-identical to the base protocol's 6 h slice
    (n compared against results/{o1_repro,origin2}/slice_stats_
    temporal.csv) and its dECE must reproduce the base protocol's
    per-cell dECE (same files' metrics_temporal.csv) to float
    precision;
  * full24's dECE must reproduce the sweep h24 run's per-cell dECE;
  * raw bucket ECE must match both the base and the h24 runs;
  * train_X / train_y md5 of the h24 dir must equal the h12 dir's.

Verdict logic: if mean dECE / harm worsens monotonically with age at
fixed 6 h span, staleness is supported as the 24 h-rebound explanation;
if flat, age alone does not explain the rebound.

Fit/apply wall-clock is logged per fit (time.perf_counter around fit
and transform separately; cost_probe convention; same machine).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from src.calibration import expected_calibration_error, fit_recalibrator
from src.data_pipeline import load_split, prepare_arm
from src.data_prep import FEATURE_COLS, VocabEncoder
from src.models import MODEL_NAMES
from src.run_experiments import load_config
from src.train_eval import train_model

ECE_BINS = 15
GAPS = (1, 2, 3)
WINDOWS = {  # condition -> (age_hours, hod_lo, hod_hi inclusive)
    "age0": (0, 18, 23),
    "age6": (6, 12, 17),
    "age12": (12, 6, 11),
    "age18": (18, 0, 5),
}
BASE_DIR = {1: "results/o1_repro", 2: "results/origin2"}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def delta_ece_by_cell(metrics_csv: str) -> dict:
    df = pd.read_csv(metrics_csv)
    raw = df[df.variant == "raw"].set_index(["model", "bucket"])["ece"]
    out = {}
    for _, r in df[df.variant != "raw"].iterrows():
        out[(r.model, r.bucket, r.variant)] = r.ece - raw.loc[(r.model, r.bucket)]
    raw_ece = {(m, b): v for (m, b), v in raw.items()}
    return out, raw_ece


def decode_hod(arm_dir: str, X: np.ndarray) -> np.ndarray:
    """Recover hour_of_day (0-23) per row from the encoded feature matrix."""
    with open(os.path.join(arm_dir, "encoder.json")) as f:
        enc = json.load(f)
    j = FEATURE_COLS.index("hour_of_day")
    offset = int(enc["field_offsets"][j])
    vocab = enc["vocab"][j]  # position i <-> local index i+1
    local = np.asarray(X[:, j]).astype(np.int64) - offset
    assert (local >= 1).all(), "hour_of_day has UNK rows; cannot decode"
    lut = np.array([int(v) for v in vocab], dtype=np.int64)
    return lut[local - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--origin", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out-dir", default="results/recency")
    args = ap.parse_args()
    o = args.origin
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(f"config_sweep_o{o}_h24.yaml")
    seed = int(cfg["seed"])
    assert seed == 42
    cap = int(cfg["data"]["train_cap_rows"])

    # replicate run_experiments.main seeding exactly
    np.random.seed(seed)
    import torch
    torch.set_num_threads(int(cfg.get("train", {}).get("torch_threads", 2)))
    torch.manual_seed(seed)

    root24 = f"data/processed_sweep_o{o}_h24"
    manifest = prepare_arm("temporal", cfg, root24, cap)  # fingerprint reuse
    encoder = VocabEncoder.load(manifest["encoder_path"])
    arm = os.path.join(root24, "temporal")

    # --- validation references ------------------------------------------------
    base_delta, base_raw = delta_ece_by_cell(
        os.path.join(BASE_DIR[o], "metrics_temporal.csv"))
    sweep24_delta, sweep24_raw = delta_ece_by_cell(
        f"results/sweep/hours_24/origin{o}/metrics_temporal.csv")
    slice_stats = pd.read_csv(os.path.join(BASE_DIR[o], "slice_stats_temporal.csv"))
    base_n = {int(b[3:]): int(grp["n_calib"].iloc[0])
              for b, grp in slice_stats.groupby("bucket")}

    val = {"origin": o, "seed": seed,
           "train_md5_h24_eq_h12": {
               part: md5(os.path.join(arm, f"train_{part}.npy")) ==
                     md5(f"data/processed_sweep_o{o}_h12/temporal/train_{part}.npy")
               for part in ("X", "y")},
           "age0_n_eq_base_slice": {},
           "age0_vs_base_max_abs_diff": 0.0,
           "full24_vs_sweep24_max_abs_diff": 0.0,
           "raw_vs_base_max_abs_diff": 0.0,
           "raw_vs_sweep24_max_abs_diff": 0.0,
           "windows": {}}

    X_train, y_train = load_split(arm, "train")
    print(f"[recency o{o}] train rows: {len(y_train):,}; "
          f"train md5 match: {val['train_md5_h24_eq_h12']}", flush=True)

    # --- per-gap window bookkeeping -------------------------------------------
    gap_info: dict[int, dict] = {}
    for g in GAPS:
        Xc, yc = load_split(arm, f"calib_gap{g}")
        hod = decode_hod(arm, Xc)
        assert bool(np.all(np.diff(hod) >= 0)), f"hod not non-decreasing (o{o} gap{g})"
        assert set(np.unique(hod).tolist()) == set(range(24)), \
            f"24h slice does not cover all 24 hours (o{o} gap{g})"
        wins = {}
        for cond, (age, lo, hi) in WINDOWS.items():
            idx = np.flatnonzero((hod >= lo) & (hod <= hi))
            assert set(np.unique(hod[idx]).tolist()) == set(range(lo, hi + 1))
            wins[cond] = idx
        wins["full24"] = np.arange(len(yc))
        gap_info[g] = {"hod": hod, "wins": wins, "n24": int(len(yc)),
                       "win_n": {c: int(len(i)) for c, i in wins.items()}}
        val["windows"][str(g)] = {"n24": int(len(yc)),
                                  "win_n": {c: int(len(i)) for c, i in wins.items()}}
        val["age0_n_eq_base_slice"][str(g)] = bool(len(wins["age0"]) == base_n[g])
        print(f"[recency o{o}] gap{g}: n24={len(yc):,} "
              f"windows={ {c: len(i) for c, i in wins.items()} } "
              f"age0 n == base 6h n ({base_n[g]:,}): {val['age0_n_eq_base_slice'][str(g)]}",
              flush=True)

    rows = []
    for model_name in MODEL_NAMES:
        t0 = time.time()
        model = train_model(model_name, encoder, X_train, y_train, cfg, seed=seed)
        print(f"[recency o{o}] {model_name}: trained in {time.time() - t0:.1f}s", flush=True)
        for g in GAPS:
            gi = gap_info[g]
            Xc, yc = load_split(arm, f"calib_gap{g}")
            Xt, yt = load_split(arm, f"test_gap{g}")
            yc = np.asarray(yc)
            yt = np.asarray(yt)
            p_slice = model.predict_proba(Xc)
            p_test = model.predict_proba(Xt)
            ece_raw = expected_calibration_error(yt, p_test, ECE_BINS)
            key_raw = (model_name, f"gap{g}")
            val["raw_vs_base_max_abs_diff"] = max(
                val["raw_vs_base_max_abs_diff"], abs(ece_raw - base_raw[key_raw]))
            val["raw_vs_sweep24_max_abs_diff"] = max(
                val["raw_vs_sweep24_max_abs_diff"], abs(ece_raw - sweep24_raw[key_raw]))
            for cond, idx in gi["wins"].items():
                age = WINDOWS[cond][0] if cond in WINDOWS else -1
                hod_rng = (f"{WINDOWS[cond][1]}-{WINDOWS[cond][2]}"
                           if cond in WINDOWS else "0-23")
                for kind in ("platt", "isotonic"):
                    t0 = time.perf_counter()
                    recal = fit_recalibrator(kind, p_slice[idx], yc[idx])
                    fit_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    p_cal = recal.transform(p_test)
                    apply_s = time.perf_counter() - t0
                    ece_cal = expected_calibration_error(yt, p_cal, ECE_BINS)
                    d = ece_cal - ece_raw
                    rows.append({
                        "origin": o, "seed": seed, "model": model_name, "gap": g,
                        "bucket": f"gap{g}", "calibrator": kind,
                        "condition": cond, "age_hours": age, "window_hod": hod_rng,
                        "n_fit": int(len(idx)), "n_slice24": gi["n24"],
                        "n_test": int(len(yt)),
                        "ece_raw": ece_raw, "ece_cal": ece_cal, "delta_ece": d,
                        "fit_time_s": fit_s, "apply_time_s": apply_s,
                    })
                    key = (model_name, f"gap{g}", kind)
                    if cond == "age0":
                        val["age0_vs_base_max_abs_diff"] = max(
                            val["age0_vs_base_max_abs_diff"],
                            abs(d - base_delta[key]))
                    elif cond == "full24":
                        val["full24_vs_sweep24_max_abs_diff"] = max(
                            val["full24_vs_sweep24_max_abs_diff"],
                            abs(d - sweep24_delta[key]))
            print(f"[recency o{o}] {model_name} gap{g}: raw ECE {ece_raw:.4f} done", flush=True)
        del model

    out_csv = os.path.join(args.out_dir, f"recency_o{o}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    with open(os.path.join(args.out_dir, f"recency_o{o}_validation.json"), "w") as f:
        json.dump(val, f, indent=2)
    print(f"[recency o{o}] wrote {out_csv}; validation: "
          f"age0 vs base max|diff|={val['age0_vs_base_max_abs_diff']:.3e}, "
          f"full24 vs sweep24 max|diff|={val['full24_vs_sweep24_max_abs_diff']:.3e}, "
          f"raw vs base max|diff|={val['raw_vs_base_max_abs_diff']:.3e}",
          flush=True)


if __name__ == "__main__":
    main()
