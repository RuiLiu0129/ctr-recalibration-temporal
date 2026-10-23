"""End-to-end orchestration (Plan A: streaming pipeline).

Arms
----
1. Temporal arm (main): train on a stride sample (capped, see
   ``data.train_cap_rows``) of the first ``split.train_days`` days; evaluate
   on FULL day-buckets at increasing temporal distance (gaps from config).
   For each bucket: raw metrics, then Platt / isotonic recalibration fitted
   on the bucket's recent calibration slice, then the bidding simulation.
2. Random arm (replication): seeded reservoir sample split 70/10/20; the
   same evaluation on its test slice, with the validation slice as the
   recalibration set. Comparing this arm's raw test AUC/log loss against
   the temporal gap-0 bucket gives the split-inflation replication numbers.

Data flow: ``data_pipeline.prepare_arm`` streams the raw CSV twice per arm
and writes encoded splits (.npy memmaps) under ``data.processed_dir``;
training/evaluation then stream batches from those memmaps, so peak memory
stays far below the size of the dataset. Prepared splits are reused when
their config fingerprint matches (``--force-prepare`` overrides).

Outputs (under the results dir)
-------------------------------
- metrics_<arm>.csv / bidding_<arm>.csv   written as each arm finishes
- metrics.csv / bidding.csv / summary.md  merged at the end
- reliability_<arm>_<model>_<bucket>.csv  per-bin reliability data
- plots/reliability_<arm>_<model>_<bucket>.png   raw-variant diagrams

``--resume`` skips any arm whose per-arm result CSVs already exist in the
results dir (their rows are loaded back for the merge).

Usage:
    python -m src.run_experiments --config config.yaml
    python -m src.run_experiments --train-cap 500000 --results-dir results_validation
    python -m src.run_experiments --quick        # small-train smoke run
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import copy
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .bidding_sim import run_bidding_sim
from .calibration import clip_probs, fit_recalibrator, reliability_table
from .data_pipeline import load_split, prepare_arm
from .data_prep import VocabEncoder
from .models import MODEL_NAMES
from .train_eval import evaluate_predictions, train_model


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_quick_overrides(cfg: dict) -> dict:
    """Small-train smoke run on the streaming pipeline.

    Caps each arm's training set at 200k rows and trains the deep models
    for 1 epoch, into a separate processed dir. Note: unlike the legacy
    in-memory smoke run, temporal *test buckets stay full-size*, so this is
    a wiring check, not a sub-minute run.
    """
    cfg = copy.deepcopy(cfg)
    cfg["_cap_override"] = 200_000
    cfg["data"]["processed_dir"] = cfg["data"].get("processed_dir", "data/processed") + "_quick"
    for name in ("deepfm", "dcnv2"):
        cfg["models"][name]["epochs"] = 1
    cfg["models"]["lr"]["epochs"] = 1
    return cfg


def _md_table(df: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot_reliability(table: pd.DataFrame, path: str, title: str) -> None:
    filled = table.dropna(subset=["mean_pred"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    ax.plot(filled["mean_pred"], filled["frac_pos"], "o-", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical click rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _evaluate_bucket(model_name, model, X_calib, y_calib, X_test, y_test,
                     cfg, arm, bucket_label, seed, preds_dir=None):
    """Raw + recalibrated metrics, reliability data and bidding for one bucket.

    X_* may be read-only memmaps; labels are materialised (int8 on disk, so
    a full bucket costs only a few MB). Returns (metric_rows,
    reliability_df, bidding_df, slice_row).

    Tier-B instrumentation (additive; default metrics unchanged):
    ``slice_row`` evaluates the RAW model on the bucket's calibration slice
    itself (AUC / log loss / ECE / mean prediction / observed CTR / signed
    bias). The raw model never trained on that slice, so the slice ECE is a
    genuinely held-out, serving-time observable -- the quantity a
    production recalibration gate would use. Metric rows additionally carry
    mean_pred / obs_ctr / signed_bias per variant. When ``preds_dir`` is
    given, per-impression probabilities (float32) are saved there for
    downstream paired analyses (RQ4 bootstrap / calibration oracle).
    """
    ece_bins = int(cfg["calibration"]["ece_bins"])
    y_test = np.asarray(y_test)
    y_calib = np.asarray(y_calib)
    variants: dict[str, np.ndarray] = {"raw": model.predict_proba(X_test)}
    p_calib = model.predict_proba(X_calib)
    for kind in cfg["calibration"]["methods"]:
        recal = fit_recalibrator(kind, p_calib, y_calib)
        variants[kind] = recal.transform(variants["raw"])

    p_calib_c = clip_probs(p_calib)
    slice_mean_pred = float(p_calib_c.mean())
    slice_ctr = float(y_calib.mean())
    slice_row = {
        "arm": arm, "model": model_name, "bucket": bucket_label,
        "n_calib": len(y_calib), "obs_ctr": slice_ctr,
        "mean_pred": slice_mean_pred,
        "signed_bias": slice_mean_pred - slice_ctr,
        **evaluate_predictions(y_calib, p_calib, ece_bins),
    }

    test_ctr = float(y_test.mean())
    metric_rows, reliability_frames = [], []
    for variant, p in variants.items():
        m = evaluate_predictions(y_test, p, ece_bins)
        mean_pred = float(clip_probs(p).mean())
        metric_rows.append({
            "arm": arm, "model": model_name, "bucket": bucket_label,
            "variant": variant, "n_test": len(y_test), "n_calib": len(y_calib), **m,
            "mean_pred": mean_pred, "obs_ctr": test_ctr,
            "signed_bias": mean_pred - test_ctr,
        })
        rel = reliability_table(y_test, p, ece_bins)
        rel.insert(0, "variant", variant)
        reliability_frames.append(rel)
        if preds_dir is not None:
            np.save(os.path.join(
                preds_dir, f"{arm}_{model_name}_{bucket_label}_{variant}.npy"),
                p.astype(np.float32))
    if preds_dir is not None:
        np.save(os.path.join(
            preds_dir, f"{arm}_{model_name}_{bucket_label}_calib_raw.npy"),
            p_calib.astype(np.float32))

    bidding = run_bidding_sim(y_test, variants, cfg, seed=seed)
    bidding.insert(0, "bucket", bucket_label)
    bidding.insert(0, "model", model_name)
    bidding.insert(0, "arm", arm)
    return (metric_rows, pd.concat(reliability_frames, ignore_index=True),
            bidding, slice_row)


# ---------------------------------------------------------------------------
# arms (streaming over prepared memmaps)
# ---------------------------------------------------------------------------

def run_arm(arm: str, cfg: dict, manifest: dict, arm_dir: str, results_dir: str,
            save_predictions: bool = False):
    encoder = VocabEncoder.load(manifest["encoder_path"])
    seed = int(cfg["seed"])
    X_train, y_train = load_split(arm_dir, "train")
    print(f"[{arm}] train rows: {len(y_train):,}", flush=True)
    preds_dir = None
    if save_predictions:
        preds_dir = os.path.join(results_dir, "preds")
        os.makedirs(preds_dir, exist_ok=True)

    metric_rows, bidding_frames, slice_rows = [], [], []
    for model_name in MODEL_NAMES:
        t0 = time.time()
        model = train_model(model_name, encoder, X_train, y_train, cfg, seed=seed)
        print(f"[{arm}] {model_name}: trained in {time.time() - t0:.1f}s", flush=True)
        if arm == "temporal":
            for gap in [int(g) for g in cfg["split"]["test_gaps_days"]]:
                X_calib, y_calib = load_split(arm_dir, f"calib_gap{gap}")
                X_test, y_test = load_split(arm_dir, f"test_gap{gap}")
                t0 = time.time()
                rows, rel, bidding, slice_row = _evaluate_bucket(
                    model_name, model, X_calib, y_calib, X_test, y_test,
                    cfg, arm=arm, bucket_label=f"gap{gap}", seed=seed + gap,
                    preds_dir=preds_dir)
                metric_rows += rows
                bidding_frames.append(bidding)
                slice_rows.append(slice_row)
                rel.to_csv(os.path.join(
                    results_dir, f"reliability_temporal_{model_name}_gap{gap}.csv"),
                    index=False)
                raw_rel = rel[rel["variant"] == "raw"].drop(columns=["variant"])
                _plot_reliability(raw_rel, os.path.join(
                    results_dir, "plots",
                    f"reliability_temporal_{model_name}_gap{gap}.png"),
                    f"{model_name} temporal gap {gap} (raw)")
                raw_row = rows[0]
                print(f"[{arm}] {model_name} gap{gap}: AUC {raw_row['auc']:.4f} "
                      f"ECE {raw_row['ece']:.4f} ({time.time() - t0:.1f}s)", flush=True)
        else:
            X_val, y_val = load_split(arm_dir, "val")
            X_test, y_test = load_split(arm_dir, "test")
            t0 = time.time()
            rows, rel, bidding, slice_row = _evaluate_bucket(
                model_name, model, X_val, y_val, X_test, y_test,
                cfg, arm=arm, bucket_label="test", seed=seed,
                preds_dir=preds_dir)
            metric_rows += rows
            bidding_frames.append(bidding)
            slice_rows.append(slice_row)
            rel.to_csv(os.path.join(
                results_dir, f"reliability_random_{model_name}_test.csv"), index=False)
            raw_rel = rel[rel["variant"] == "raw"].drop(columns=["variant"])
            _plot_reliability(raw_rel, os.path.join(
                results_dir, "plots", f"reliability_random_{model_name}_test.png"),
                f"{model_name} random split (raw)")
            raw_row = rows[0]
            print(f"[{arm}] {model_name} test: AUC {raw_row['auc']:.4f} "
                  f"ECE {raw_row['ece']:.4f} ({time.time() - t0:.1f}s)", flush=True)
        del model
    return metric_rows, bidding_frames, slice_rows


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def write_summary(metrics: pd.DataFrame, bidding: pd.DataFrame, path: str) -> None:
    raw = metrics[metrics["variant"] == "raw"]
    repl = raw[((raw["arm"] == "random") & (raw["bucket"] == "test")) |
               ((raw["arm"] == "temporal") & (raw["bucket"] == "gap0"))]
    repl = repl[["model", "arm", "auc", "logloss", "ece"]].sort_values(["model", "arm"])

    ece = metrics[metrics["arm"] == "temporal"].pivot_table(
        index=["model", "bucket"], columns="variant", values="ece").reset_index()

    bid = bidding[bidding["arm"] == "temporal"].pivot_table(
        index=["model", "bucket"], columns="strategy", values="profit").reset_index()

    with open(path, "w") as f:
        f.write("# Summary — Calibration under temporal splits (Avazu)\n\n")
        f.write("## RQ1 replication: random split vs temporal gap-0 (raw model)\n\n")
        f.write(_md_table(repl) + "\n\n")
        f.write("## RQ2/RQ3: ECE by temporal distance and variant "
                "(platt / isotonic fitted per bucket on the recent slice)\n\n")
        f.write(_md_table(ece) + "\n\n")
        f.write("## RQ4: simulated bidding profit by strategy (temporal arm)\n\n")
        f.write(_md_table(bid) + "\n")
        f.write("\nOracle is a ground-truth upper bound, not a deployable strategy.\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: 200k train rows/arm, 1 epoch, separate processed dir")
    parser.add_argument("--train-cap", type=int, default=None,
                        help="override the per-arm training-row cap "
                             "(default: data.train_cap_rows from the config)")
    parser.add_argument("--processed-dir", default=None,
                        help="override data.processed_dir (prepared .npy splits)")
    parser.add_argument("--results-dir", default=None,
                        help="override results_dir from the config")
    parser.add_argument("--arm", choices=("both", "temporal", "random"), default="both")
    parser.add_argument("--resume", action="store_true",
                        help="skip arms whose per-arm result CSVs already exist")
    parser.add_argument("--save-predictions", action="store_true",
                        help="save per-impression probabilities (float32 .npy) "
                             "under <results-dir>/preds for paired analyses")
    parser.add_argument("--force-prepare", action="store_true",
                        help="re-run data preparation even if fingerprints match")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick_overrides(cfg)
    np.random.seed(int(cfg["seed"]))

    try:
        import torch
        torch.set_num_threads(int(cfg.get("train", {}).get("torch_threads", 2)))
        torch.manual_seed(int(cfg["seed"]))  # deterministic deep-model inits
    except Exception:
        pass

    cap = args.train_cap
    if cap is None:
        cap = cfg.get("_cap_override", cfg["data"].get("train_cap_rows"))
    processed_root = args.processed_dir or cfg["data"].get("processed_dir", "data/processed")
    results_dir = args.results_dir or cfg.get("results_dir", "results")
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    print(f"train cap per arm: {cap if cap is not None else 'none'}; "
          f"processed dir: {processed_root}; results dir: {results_dir}", flush=True)

    arms = ("temporal", "random") if args.arm == "both" else (args.arm,)
    metric_frames, bidding_frames = [], []
    for arm in arms:
        m_path = os.path.join(results_dir, f"metrics_{arm}.csv")
        b_path = os.path.join(results_dir, f"bidding_{arm}.csv")
        s_path = os.path.join(results_dir, f"slice_stats_{arm}.csv")
        if args.resume and os.path.exists(m_path) and os.path.exists(b_path):
            print(f"[{arm}] --resume: loading existing per-arm results", flush=True)
            metric_frames.append(pd.read_csv(m_path))
            bidding_frames.append(pd.read_csv(b_path))
            continue
        t0 = time.time()
        manifest = prepare_arm(arm, cfg, processed_root, cap, force=args.force_prepare)
        print(f"[{arm}] prepare done in {time.time() - t0:.1f}s", flush=True)
        t0 = time.time()
        rows, bids, slice_rows = run_arm(
            arm, cfg, manifest, os.path.join(processed_root, arm), results_dir,
            save_predictions=args.save_predictions)
        print(f"[{arm}] train+eval done in {time.time() - t0:.1f}s", flush=True)
        arm_metrics = pd.DataFrame(rows)
        arm_bidding = pd.concat(bids, ignore_index=True)
        arm_metrics.to_csv(m_path, index=False)
        arm_bidding.to_csv(b_path, index=False)
        if slice_rows:
            pd.DataFrame(slice_rows).to_csv(s_path, index=False)
        print(f"[{arm}] wrote {m_path} and {b_path}", flush=True)
        metric_frames.append(arm_metrics)
        bidding_frames.append(arm_bidding)

    metrics = pd.concat(metric_frames, ignore_index=True)
    bidding = pd.concat(bidding_frames, ignore_index=True)
    metrics.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    bidding.to_csv(os.path.join(results_dir, "bidding.csv"), index=False)
    write_summary(metrics, bidding, os.path.join(results_dir, "summary.md"))
    print(f"done. Results in {results_dir}/ (metrics.csv, bidding.csv, summary.md, "
          "metrics_<arm>.csv, bidding_<arm>.csv, reliability_*.csv, plots/)")


if __name__ == "__main__":
    main()
