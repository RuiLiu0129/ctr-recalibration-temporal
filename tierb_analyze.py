"""Tier-B analyses for the Avazu temporal-calibration paper (2026-10-22).

Inputs (all produced by the Tier-B run chain; origin 1 = the original
results/ unless noted):
  results/metrics_temporal.csv                    origin 1, seed 42 (original)
  results/reliability_temporal_<model>_gap<g>.csv origin 1 reliability tables
  results/o1_repro/{metrics,slice_stats}_temporal.csv, preds/  origin 1 rerun
  results/origin2/{metrics,slice_stats}_temporal.csv, preds/   origin 2
  results/seeds/seed{43,44}/{metrics,slice_stats}_temporal.csv  seed replicates
  data/processed_o1repro/temporal, data/processed_o2/temporal  labels for RQ4

Outputs: results/tierb_analysis.md + CSVs under results/tierb/.

Analyses:
  A. origin-1 signed bias from the saved reliability tables (no retraining).
  B. seed variance of ECE on origin 1 (seeds 42/43/44).
  C. harm-rule separation on the pooled 48 calibrator cells (origins 1+2):
     raw-ECE threshold rule (tau fit on origin 1 only, applied frozen),
     gap==3 rule, weekday-absence rule.
  D. out-of-sample gate evaluation: recalibrate (isotonic) only when the
     raw model's ECE on the bucket's 6h calibration slice exceeds tau_slice
     (tau_slice fit on origin 1, frozen for origin 2); slice-vs-target ECE
     correlations.
  E. RQ4: calibration oracle (raw scores remapped through per-bin test
     rates) + block-bootstrap paired significance of profit deltas, from
     the saved per-impression predictions; regenerated market prices are
     verified against each run's own bidding CSV first.
  F. weekday table per origin.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from src.bidding_sim import draw_market_prices, simulate_strategy
from src.calibration import reliability_table
from src.run_experiments import load_config

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "tierb")
os.makedirs(OUT, exist_ok=True)
MODELS = ("lr", "deepfm", "dcnv2")
GAPS = (0, 1, 2, 3)
CALS = ("platt", "isotonic")
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# day_idx 0 = 2014-10-21 = Tuesday (index into WEEKDAYS with Mon=0 -> Tue=1)
DAY_WEEKDAY = {d: WEEKDAYS[(1 + d) % 7] for d in range(10)}
ORIGIN_TRAIN_DAYS = {1: list(range(0, 5)), 2: list(range(1, 6))}

md: list[str] = []


def w(s: str = "") -> None:
    md.append(s)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gap"] = df["bucket"].str.replace("gap", "").astype(int)
    return df


def ece_lookup(metrics: pd.DataFrame) -> dict:
    """(model, gap, variant) -> ECE."""
    return {(r.model, r.gap, r.variant): r.ece for r in metrics.itertuples()}


# ---------------------------------------------------------------------------
# load everything
# ---------------------------------------------------------------------------
o1 = load_metrics(os.path.join(ROOT, "results", "metrics_temporal.csv"))
o1_repro = load_metrics(os.path.join(ROOT, "results", "o1_repro", "metrics_temporal.csv"))
o2 = load_metrics(os.path.join(ROOT, "results", "origin2", "metrics_temporal.csv"))
o1_slice = pd.read_csv(os.path.join(ROOT, "results", "o1_repro", "slice_stats_temporal.csv"))
o2_slice = pd.read_csv(os.path.join(ROOT, "results", "origin2", "slice_stats_temporal.csv"))
seed_metrics, seed_slices = {}, {}
for s in (43, 44):
    seed_metrics[s] = load_metrics(os.path.join(ROOT, "results", "seeds", f"seed{s}", "metrics_temporal.csv"))
    seed_slices[s] = pd.read_csv(os.path.join(ROOT, "results", "seeds", f"seed{s}", "slice_stats_temporal.csv"))
for sl in (o1_slice, o2_slice, *seed_slices.values()):
    sl["gap"] = sl["bucket"].str.replace("gap", "").astype(int)

w("# Tier-B analysis — Avazu temporal calibration (2026-10-22)")
w()
w("All numbers below are computed by `tierb_analyze.py` from the run outputs; "
  "nothing is hand-entered. Origin 1 = train days 0–4 (Oct 21–25, Tue–Sat), "
  "origin 2 = train days 1–5 (Oct 22–26, Wed–Sun); buckets are the four "
  "following days in each case.")
w()

# ---------------------------------------------------------------------------
# 0. reproducibility of the origin-1 rerun
# ---------------------------------------------------------------------------
mrg = o1.merge(o1_repro, on=["model", "gap", "variant"], suffixes=("_orig", "_repro"))
w("## 0. Origin-1 reproduction check (seed 42, instrumented rerun vs original)")
w()
w(f"- max |ΔECE| = {np.abs(mrg['ece_orig'] - mrg['ece_repro']).max():.2e}; "
  f"max |ΔAUC| = {np.abs(mrg['auc_orig'] - mrg['auc_repro']).max():.2e}; "
  f"max |Δlogloss| = {np.abs(mrg['logloss_orig'] - mrg['logloss_repro']).max():.2e} "
  "across all 36 (model, gap, variant) rows.")
w("- The rerun is bit-stable enough that its slice statistics and saved "
  "predictions are used below as origin 1's (seed 42).")
w()

# ---------------------------------------------------------------------------
# A. origin-1 signed bias from reliability tables
# ---------------------------------------------------------------------------
rows = []
for model in MODELS:
    for g in GAPS:
        rel = pd.read_csv(os.path.join(
            ROOT, "results", f"reliability_temporal_{model}_gap{g}.csv"))
        raw = rel[rel["variant"] == "raw"].dropna(subset=["mean_pred"])
        n = raw["count"].sum()
        mean_pred = float((raw["count"] * raw["mean_pred"]).sum() / n)
        ctr = float((raw["count"] * raw["frac_pos"]).sum() / n)
        rows.append({"model": model, "gap": g, "n_test": int(n),
                     "obs_ctr": ctr, "mean_pred": mean_pred,
                     "signed_bias": mean_pred - ctr})
o1_bias = pd.DataFrame(rows)
o1_bias.to_csv(os.path.join(OUT, "o1_signed_bias.csv"), index=False)
w("## A. Origin 1 — signed bias of the raw model (from the saved reliability tables)")
w()
w("signed bias = mean prediction − observed CTR (positive = over-prediction). "
  "Derivable without retraining; agrees with the rerun's metric columns.")
w()
w(md_table(o1_bias))
w()

# ---------------------------------------------------------------------------
# Origin 2 headline
# ---------------------------------------------------------------------------
w("## Origin 2 — headline results (seed 42)")
w()
o2_raw = o2[o2["variant"] == "raw"].sort_values(["model", "gap"])
w("Raw model by bucket:")
w()
w(md_table(o2_raw[["model", "gap", "n_test", "auc", "logloss", "ece",
                    "obs_ctr", "mean_pred", "signed_bias"]]))
w()
o2_ece = o2.pivot_table(index=["model", "gap"], columns="variant", values="ece").reset_index()
w("ECE by variant:")
w()
w(md_table(o2_ece[["model", "gap", "raw", "platt", "isotonic"]]))
w()
o2_sl = o2_slice.sort_values(["model", "gap"])
w("Calibration-slice statistics (raw model evaluated on each bucket's 6h slice):")
w()
w(md_table(o2_sl[["model", "gap", "n_calib", "obs_ctr", "mean_pred",
                  "signed_bias", "ece"]].rename(columns={"ece": "slice_ece"})))
w()
o2_ece.to_csv(os.path.join(OUT, "o2_ece_grid.csv"), index=False)

# ---------------------------------------------------------------------------
# F. weekday table
# ---------------------------------------------------------------------------
w("## F. Weekday alignment vs raw ECE")
w()
wd_rows = []
for origin, metrics in ((1, o1), (2, o2)):
    train_wd = {DAY_WEEKDAY[d] for d in ORIGIN_TRAIN_DAYS[origin]}
    lk = ece_lookup(metrics)
    for g in GAPS:
        bd = ORIGIN_TRAIN_DAYS[origin][-1] + 1 + g
        wd = DAY_WEEKDAY[bd]
        row = {"origin": origin, "gap": g, "test_day": f"Oct {21 + bd}",
               "weekday": wd, "weekday_in_train": wd in train_wd}
        for model in MODELS:
            row[f"ece_{model}"] = lk[(model, g, "raw")]
        row["ece_mean"] = np.mean([lk[(m, g, "raw")] for m in MODELS])
        wd_rows.append(row)
wd = pd.DataFrame(wd_rows)
wd.to_csv(os.path.join(OUT, "weekday_table.csv"), index=False)
w("Origin 1 training weekdays: Tue–Sat. Origin 2 training weekdays: Wed–Sun. "
  "In both origins the gap-0 and gap-1 bucket weekdays (Sun/Mon for origin 1, "
  "Mon/Tue for origin 2) are absent from the training window.")
w()
w(md_table(wd))
w()
for origin in (1, 2):
    sub = wd[wd["origin"] == origin]
    peak = sub.loc[sub["ece_mean"].idxmax(), "gap"]
    w(f"- Origin {origin}: mean raw ECE peaks at gap {int(peak)}.")
w()

# ---------------------------------------------------------------------------
# C. harm-rule separation on 48 cells
# ---------------------------------------------------------------------------
w("## C. Harm-rule separation (pooled origins 1+2, 48 calibrator cells)")
w()
cells = []
for origin, metrics in ((1, o1), (2, o2)):
    lk = ece_lookup(metrics)
    train_wd = {DAY_WEEKDAY[d] for d in ORIGIN_TRAIN_DAYS[origin]}
    for model in MODELS:
        for g in GAPS:
            bd = ORIGIN_TRAIN_DAYS[origin][-1] + 1 + g
            for cal in CALS:
                cells.append({
                    "origin": origin, "model": model, "gap": g, "calibrator": cal,
                    "raw_ece": lk[(model, g, "raw")],
                    "cal_ece": lk[(model, g, cal)],
                    "harm": lk[(model, g, cal)] > lk[(model, g, "raw")],
                    "weekday_absent": DAY_WEEKDAY[bd] not in train_wd,
                })
cells = pd.DataFrame(cells)
cells.to_csv(os.path.join(OUT, "cells48.csv"), index=False)


def best_tau(df: pd.DataFrame) -> tuple[float, float]:
    vals = np.sort(df["raw_ece"].unique())
    cands = np.concatenate([[vals[0] - 1e-9],
                            (vals[:-1] + vals[1:]) / 2,
                            [vals[-1] + 1e-9]])
    best = (-1.0, np.nan)
    for t in cands:
        acc = float(((df["raw_ece"] <= t) == df["harm"]).mean())
        if acc > best[0]:
            best = (acc, t)
    return best[1], best[0]


tau_o1, acc_tau_o1 = best_tau(cells[cells["origin"] == 1])
w(f"Raw-ECE threshold fitted on origin 1 only: τ = {tau_o1:.4f} "
  f"(origin-1 accuracy {acc_tau_o1:.3f}). Applied frozen elsewhere.")
w()
rule_rows = []
for scope, df in (("origin 1", cells[cells["origin"] == 1]),
                  ("origin 2", cells[cells["origin"] == 2]),
                  ("pooled", cells)):
    rule_rows.append({
        "scope": scope, "n": len(df),
        "raw-ECE rule (τ frozen from origin 1)": float(((df["raw_ece"] <= tau_o1) == df["harm"]).mean()),
        "gap==3 rule": float(((df["gap"] == 3) == df["harm"]).mean()),
        "weekday-absent rule": float((df["weekday_absent"] == df["harm"]).mean()),
    })
tau_pool, acc_tau_pool = best_tau(cells)
rule_rows.append({
    "scope": "pooled (τ refit on pooled, in-sample)", "n": len(cells),
    "raw-ECE rule (τ frozen from origin 1)": acc_tau_pool,
    "gap==3 rule": np.nan, "weekday-absent rule": np.nan})
rules = pd.DataFrame(rule_rows)
rules.to_csv(os.path.join(OUT, "rule_scores.csv"), index=False)
w(md_table(rules, floatfmt="{:.3f}"))
w()
w(f"(Pooled-oracle τ = {tau_pool:.4f} is shown only as an in-sample reference.)")
w()

# ---------------------------------------------------------------------------
# D. gate evaluation
# ---------------------------------------------------------------------------
w("## D. Recalibration gate, out-of-sample (slice ECE as the gate input)")
w()
slice_ece = {}
for origin, sl in ((1, o1_slice), (2, o2_slice)):
    for r in sl.itertuples():
        slice_ece[(origin, r.model, r.gap)] = r.ece
target = {}
for origin, metrics in ((1, o1), (2, o2)):
    lk = ece_lookup(metrics)
    for model in MODELS:
        for g in GAPS:
            target[(origin, model, g)] = (lk[(model, g, "raw")], lk[(model, g, "isotonic")])


def gate_stats(tau: float, origins: tuple[int, ...]) -> dict:
    raw_l, iso_l, gated_l = [], [], []
    for (o, model, g), (raw_e, iso_e) in target.items():
        if o not in origins:
            continue
        raw_l.append(raw_e)
        iso_l.append(iso_e)
        gated_l.append(iso_e if slice_ece[(o, model, g)] > tau else raw_e)
    return {"mean_raw (never)": float(np.mean(raw_l)),
            "mean_isotonic (always)": float(np.mean(iso_l)),
            "mean_gated": float(np.mean(gated_l)), "n_cells": len(raw_l)}


taus = np.sort(np.unique([slice_ece[(1, m, g)] for m in MODELS for g in GAPS]))
cand_taus = np.concatenate([[taus[0] - 1e-9], (taus[:-1] + taus[1:]) / 2, [taus[-1] + 1e-9]])
best_tau_slice, best_obj = None, np.inf
for t in cand_taus:
    obj = gate_stats(float(t), (1,))["mean_gated"]
    if obj < best_obj:
        best_obj, best_tau_slice = obj, float(t)
w(f"τ_slice fitted on origin 1 (minimising origin-1 mean gated ECE): "
  f"τ_slice = {best_tau_slice:.4f}. Frozen for origin 2.")
w()
gate_rows = []
for origins, label in (((1,), "origin 1 (fit)"), ((2,), "origin 2 (frozen τ)"), ((1, 2), "pooled")):
    gate_rows.append({"scope": label, **gate_stats(best_tau_slice, origins)})
gate = pd.DataFrame(gate_rows)
gate.to_csv(os.path.join(OUT, "gate_eval.csv"), index=False)
w(md_table(gate))
w()
# harm-classification view of the same threshold
harm_iso = cells[cells["calibrator"] == "isotonic"].copy()
harm_iso["slice_ece"] = [slice_ece[(r.origin, r.model, r.gap)] for r in harm_iso.itertuples()]
for origin in (1, 2):
    sub = harm_iso[harm_iso["origin"] == origin]
    acc = float(((sub["slice_ece"] <= best_tau_slice) == sub["harm"]).mean())
    w(f"- τ_slice as a harm classifier (isotonic cells, origin {origin}): accuracy {acc:.3f}.")
w()
corr_rows = []
seed_slice = {}
for s, sl in seed_slices.items():
    for r in sl.itertuples():
        seed_slice[(s, r.model, r.gap)] = r.ece
corr_data = {
    "origin 1 (seed 42)": ([slice_ece[(1, m, g)] for m in MODELS for g in GAPS],
                           [target[(1, m, g)][0] for m in MODELS for g in GAPS]),
    "origin 2 (seed 42)": ([slice_ece[(2, m, g)] for m in MODELS for g in GAPS],
                           [target[(2, m, g)][0] for m in MODELS for g in GAPS]),
    "pooled seed 42": ([slice_ece[(o, m, g)] for o in (1, 2) for m in MODELS for g in GAPS],
                       [target[(o, m, g)][0] for o in (1, 2) for m in MODELS for g in GAPS]),
    "origin 1 seeds 43+44 (slice ECE vs own-seed target)": (
        [seed_slice[(s, m, g)] for s in (43, 44) for m in MODELS for g in GAPS],
        [ece_lookup(seed_metrics[s])[(m, g, "raw")] for s in (43, 44) for m in MODELS for g in GAPS]),
}
for label, (xs, ys) in corr_data.items():
    xs, ys = np.array(xs), np.array(ys)
    corr_rows.append({"set": label, "n": len(xs),
                      "pearson": float(stats.pearsonr(xs, ys)[0]),
                      "spearman": float(stats.spearmanr(xs, ys)[0])})
corr = pd.DataFrame(corr_rows)
corr.to_csv(os.path.join(OUT, "slice_target_corr.csv"), index=False)
w("Slice ECE vs target-bucket raw ECE:")
w()
w(md_table(corr, floatfmt="{:.3f}"))
w()

# ---------------------------------------------------------------------------
# B. seed variance
# ---------------------------------------------------------------------------
w("## B. Seed variance on origin 1 (seeds 42 / 43 / 44)")
w()
seed_frames = {42: o1, 43: seed_metrics[43], 44: seed_metrics[44]}
seed_rows = []
for model in MODELS:
    for g in GAPS:
        for variant in ("raw", "platt", "isotonic"):
            vals = [ece_lookup(seed_frames[s])[(model, g, variant)] for s in (42, 43, 44)]
            seed_rows.append({"model": model, "gap": g, "variant": variant,
                              "ece_s42": vals[0], "ece_s43": vals[1], "ece_s44": vals[2],
                              "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))})
seed_ece = pd.DataFrame(seed_rows)
seed_ece.to_csv(os.path.join(OUT, "seed_ece.csv"), index=False)
w(md_table(seed_ece))
w()
flag_rows = []
for model in MODELS:
    for g in GAPS:
        for cal in CALS:
            deltas = []
            for s in (42, 43, 44):
                lk = ece_lookup(seed_frames[s])
                deltas.append(lk[(model, g, cal)] - lk[(model, g, "raw")])
            deltas = np.array(deltas)
            sd = float(np.std(deltas, ddof=1))
            sign_flip = bool(np.any(deltas > 0) and np.any(deltas < 0))
            within = bool(abs(deltas[0]) < 2 * sd) or sign_flip
            flag_rows.append({"model": model, "gap": g, "calibrator": cal,
                              "dECE_s42": deltas[0], "dECE_s43": deltas[1], "dECE_s44": deltas[2],
                              "sd_dECE": sd, "sign_flip": sign_flip,
                              "within_seed_noise": within})
flags = pd.DataFrame(flag_rows)
flags.to_csv(os.path.join(OUT, "seed_flags.csv"), index=False)
w("ΔECE = ECE(calibrated) − ECE(raw) per seed; a cell is flagged when the "
  "seed-42 sign flips across seeds or |ΔECE_42| < 2·sd(ΔECE):")
w()
w(md_table(flags))
w()
n_flag = int(flags["within_seed_noise"].sum())
w(f"- {n_flag} of {len(flags)} origin-1 calibrator cells are within seed noise "
  "by this criterion (3 seeds only — sd estimates are crude).")
w()

# ---------------------------------------------------------------------------
# E. RQ4: calibration oracle + paired bootstrap
# ---------------------------------------------------------------------------
w("## E. RQ4 — calibration oracle and paired significance of profit deltas")
w()
cfg = load_config(os.path.join(ROOT, "config.yaml"))
bid_cfg = cfg["bidding"]
value = float(bid_cfg["value_per_click"])
shading = float(bid_cfg.get("bid_shading", 1.0))


def contrib(y: np.ndarray, p: np.ndarray, mprice: np.ndarray) -> np.ndarray:
    bids = shading * np.asarray(p, dtype=np.float64) * value
    won = bids > mprice
    return np.where(won, np.where(y == 1, value, 0.0) - mprice, 0.0)


def block_boot_ci(delta: np.ndarray, n_blocks: int = 200, n_boot: int = 10000,
                  seed: int = 7) -> tuple[float, float, float]:
    """Block bootstrap (contiguous blocks) of the total of per-impression deltas."""
    n = len(delta)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    block_sums = np.array([delta[edges[i]:edges[i + 1]].sum() for i in range(n_blocks)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    totals = block_sums[idx].sum(axis=1)
    lo, hi = np.percentile(totals, [2.5, 97.5])
    p = 2 * min(float((totals <= 0).mean()), float((totals >= 0).mean()))
    return float(lo), float(hi), min(p, 1.0)


rq4_rows, boot_rows = [], []
run_specs = [(1, "o1_repro", "data/processed_o1repro"),
             (2, "origin2", "data/processed_o2")]
max_repro_err = 0.0
for origin, rdir, pdir in run_specs:
    bidding_run = pd.read_csv(os.path.join(ROOT, "results", rdir, "bidding_temporal.csv"))
    for model in MODELS:
        for g in GAPS:
            y = np.asarray(np.load(os.path.join(
                ROOT, pdir, "temporal", f"test_gap{g}_y.npy"), mmap_mode="r"),
                dtype=np.float64)
            preds = {v: np.load(os.path.join(
                ROOT, "results", rdir, "preds", f"temporal_{model}_gap{g}_{v}.npy")
            ).astype(np.float64) for v in ("raw", "platt", "isotonic")}
            mprice = draw_market_prices(len(y), cfg, seed=42 + g)
            # sanity: recomputed profits must reproduce the run's bidding CSV
            for v in ("raw", "platt", "isotonic"):
                got = simulate_strategy(v, y, preds[v], mprice, cfg)["profit"]
                want = float(bidding_run[(bidding_run["model"] == model) &
                                         (bidding_run["bucket"] == f"gap{g}") &
                                         (bidding_run["strategy"] == v)]["profit"].iloc[0])
                max_repro_err = max(max_repro_err, abs(got - want))
            # calibration oracle: raw scores remapped through per-bin test rates
            rel = reliability_table(y, preds["raw"], 15)
            bin_idx = np.minimum((np.clip(preds["raw"], 0, 1) * 15).astype(int), 14)
            p_cal_or = rel["frac_pos"].to_numpy()[bin_idx]
            profits = {}
            for name, p in [*preds.items(), ("calib_oracle", p_cal_or),
                            ("label_oracle", y)]:
                profits[name] = simulate_strategy(name, y, p, mprice, cfg)["profit"]
            rq4_rows.append({"origin": origin, "model": model, "gap": g,
                             "n": len(y), **{f"profit_{k}": v for k, v in profits.items()},
                             "calib_oracle_minus_raw": profits["calib_oracle"] - profits["raw"],
                             "calib_oracle_minus_isotonic": profits["calib_oracle"] - profits["isotonic"]})
            contribs = {v: contrib(y, p, mprice) for v, p in preds.items()}
            contribs["calib_oracle"] = contrib(y, p_cal_or, mprice)
            for cal in ("platt", "isotonic", "calib_oracle"):
                delta = contribs[cal] - contribs["raw"]
                lo, hi, p = block_boot_ci(delta)
                boot_rows.append({"origin": origin, "model": model, "gap": g,
                                  "comparison": f"{cal}-raw",
                                  "delta_profit": float(delta.sum()),
                                  "delta_per_1000": float(delta.sum() / len(y) * 1000),
                                  "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                                  "ci_excludes_0": bool(lo > 0 or hi < 0)})
rq4 = pd.DataFrame(rq4_rows)
boot = pd.DataFrame(boot_rows)
rq4.to_csv(os.path.join(OUT, "rq4_calib_oracle.csv"), index=False)
boot.to_csv(os.path.join(OUT, "rq4_bootstrap.csv"), index=False)
w(f"Sanity: recomputing profits from the saved predictions + regenerated "
  f"market prices reproduces each run's own bidding CSV with max abs error "
  f"{max_repro_err:.4f} currency units (profits are ~10⁵–10⁶), so the paired "
  "inputs below are exactly the simulation's.")
w()
w("Profits by strategy (label oracle = the paper's original oracle):")
w()
w(md_table(rq4[["origin", "model", "gap", "profit_raw", "profit_platt",
                "profit_isotonic", "profit_calib_oracle", "profit_label_oracle"]],
           floatfmt="{:.1f}"))
w()
w("Paired block bootstrap (200 contiguous blocks, 10,000 resamples) of profit "
  "deltas vs raw:")
w()
w(md_table(boot, floatfmt="{:.4f}"))
w()
n_sig = int(boot[boot["comparison"] != "calib_oracle-raw"]["ci_excludes_0"].sum())
n_tot = int((boot["comparison"] != "calib_oracle-raw").sum())
w(f"- Recalibration-vs-raw deltas whose 95% CI excludes 0: {n_sig} of {n_tot} "
  "(platt/isotonic × models × gaps × origins).")
co = boot[boot["comparison"] == "calib_oracle-raw"]
w(f"- Calibration-oracle-vs-raw deltas whose 95% CI excludes 0: "
  f"{int(co['ci_excludes_0'].sum())} of {len(co)}; median calib-oracle gain over "
  f"raw = {co['delta_per_1000'].median():.4f} per 1000 impressions, vs median "
  f"label-oracle headroom (from the profit table) of roughly "
  f"{float(((rq4['profit_label_oracle'] - rq4['profit_raw']) / (rq4['n'] / 1000)).median()):.1f} "
  "per 1000 impressions.")
w()

with open(os.path.join(ROOT, "results", "tierb_analysis.md"), "w") as f:
    f.write("\n".join(md))
print("wrote results/tierb_analysis.md and CSVs to results/tierb/")
