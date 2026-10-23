"""Tier-C workstream 5: isotonic-on-test-day oracle for RQ4 (2026-10-22).

For each origin (o1_repro + origin2 saved predictions), model and bucket:
fit the project's isotonic recalibrator on the bucket's OWN test
impressions (raw scores -> test labels), remap the raw scores through
it, and bid with the remapped probabilities in the same lognormal-market
simulation as the runs themselves. This is the optimal monotone
recalibration achievable on test-day data -- the stronger oracle that the
Tier-B per-bin-remap calibration oracle was criticised for not being
(the bin remap destroys within-bin ordering; the isotonic remap is
monotone in the raw score and fitted on the full test distribution).

Profit deltas vs raw get the same paired block bootstrap as Tier-B
section E (200 contiguous blocks, 10,000 resamples, rng seed 7), and the
paired market prices / contribution machinery are reused verbatim so the
sanity check (recomputed profits vs each run's own bidding CSV) must
reproduce Tier-B's bit-exact result.

Outputs: results/tierc/rq4_isotonic_oracle.csv + a section appended to
results/tierc_analysis.md.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.bidding_sim import draw_market_prices, simulate_strategy
from src.calibration import IsotonicRecalibrator
from src.run_experiments import load_config

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "tierc")
os.makedirs(OUT, exist_ok=True)
MODELS = ("lr", "deepfm", "dcnv2")
GAPS = (0, 1, 2, 3)


def contrib(y: np.ndarray, p: np.ndarray, mprice: np.ndarray,
            value: float, shading: float) -> np.ndarray:
    bids = shading * np.asarray(p, dtype=np.float64) * value
    won = bids > mprice
    return np.where(won, np.where(y == 1, value, 0.0) - mprice, 0.0)


def block_boot_ci(delta: np.ndarray, n_blocks: int = 200, n_boot: int = 10000,
                  seed: int = 7) -> tuple[float, float, float]:
    """Block bootstrap (contiguous blocks) of the total of per-impression
    deltas -- verbatim from tierb_analyze.py."""
    n = len(delta)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    block_sums = np.array([delta[edges[i]:edges[i + 1]].sum() for i in range(n_blocks)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_boot, n_blocks))
    totals = block_sums[idx].sum(axis=1)
    lo, hi = np.percentile(totals, [2.5, 97.5])
    p = 2 * min(float((totals <= 0).mean()), float((totals >= 0).mean()))
    return float(lo), float(hi), min(p, 1.0)


RUNS = [(1, "o1_repro", "data/processed_o1repro", "config.yaml"),
        (2, "origin2", "data/processed_o2", "config_tierb_o2.yaml")]

rows = []
max_repro_err = 0.0
for origin, rdir, pdir, cfg_name in RUNS:
    cfg = load_config(os.path.join(ROOT, cfg_name))
    bid_cfg = cfg["bidding"]
    value = float(bid_cfg["value_per_click"])
    shading = float(bid_cfg.get("bid_shading", 1.0))
    bidding_run = pd.read_csv(os.path.join(ROOT, "results", rdir, "bidding_temporal.csv"))
    for model in MODELS:
        for g in GAPS:
            y = np.asarray(np.load(os.path.join(
                ROOT, pdir, "temporal", f"test_gap{g}_y.npy"), mmap_mode="r"),
                dtype=np.float64)
            preds = {v: np.load(os.path.join(
                ROOT, "results", rdir, "preds",
                f"temporal_{model}_gap{g}_{v}.npy")).astype(np.float64)
                for v in ("raw", "platt", "isotonic")}
            mprice = draw_market_prices(len(y), cfg, seed=42 + g)
            # sanity: recomputed slice-strategy profits vs the run's own CSV
            for v in ("raw", "platt", "isotonic"):
                got = simulate_strategy(v, y, preds[v], mprice, cfg)["profit"]
                want = float(bidding_run[(bidding_run["model"] == model) &
                                         (bidding_run["bucket"] == f"gap{g}") &
                                         (bidding_run["strategy"] == v)]["profit"].iloc[0])
                max_repro_err = max(max_repro_err, abs(got - want))
            # isotonic oracle: fit on the bucket's own test impressions
            iso = IsotonicRecalibrator().fit(preds["raw"], y)
            p_or = iso.transform(preds["raw"])
            profit_raw = simulate_strategy("raw", y, preds["raw"], mprice, cfg)["profit"]
            profit_or = simulate_strategy("isotonic_oracle", y, p_or, mprice, cfg)["profit"]
            delta = contrib(y, p_or, mprice, value, shading) - \
                contrib(y, preds["raw"], mprice, value, shading)
            lo, hi, p = block_boot_ci(delta)
            rows.append({
                "origin": origin, "model": model, "gap": g, "n": len(y),
                "comparison": "isotonic_oracle-raw",
                "profit_raw": profit_raw, "profit_isotonic_oracle": profit_or,
                "delta_profit": float(delta.sum()),
                "delta_per_1000": float(delta.sum() / len(y) * 1000),
                "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                "ci_excludes_0": bool(lo > 0 or hi < 0),
            })
            print(f"origin {origin} {model} gap{g}: delta/1000 = "
                  f"{rows[-1]['delta_per_1000']:+.4f} "
                  f"CI [{lo:+.1f}, {hi:+.1f}] excl0={rows[-1]['ci_excludes_0']}",
                  flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "rq4_isotonic_oracle.csv"), index=False)

n_beats = int((df["delta_profit"] > 0).sum())
sig = df[df["ci_excludes_0"]]
n_sig_pos = int((sig["delta_profit"] > 0).sum())
n_sig_neg = int((sig["delta_profit"] < 0).sum())
med_per1000 = float(df["delta_per_1000"].median())
med_delta = float(df["delta_profit"].median())
print(f"\nsanity max repro err = {max_repro_err:.4f}")
print(f"oracle beats raw in {n_beats}/24 cells; significant: {n_sig_pos} pos, "
      f"{n_sig_neg} neg; median delta/1000 = {med_per1000:+.4f}")

# ---------------------------------------------------------------------------
# append the summary section to results/tierc_analysis.md (idempotent)
# ---------------------------------------------------------------------------
md_path = os.path.join(ROOT, "results", "tierc_analysis.md")
existing = open(md_path).read() if os.path.exists(md_path) else ""
marker = "## 5. RQ4 — isotonic-on-test-day oracle"
if marker not in existing:
    def md_table(d: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
        cols = [str(c) for c in d.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in d.iterrows():
            cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    sec = ["", marker, "",
           "The Tier-B calibration oracle remapped raw scores through the "
           "bucket's per-bin test rates, which destroys within-bin ordering. "
           "This workstream fits the project's isotonic recalibrator on each "
           "bucket's OWN test impressions (raw scores → test labels) — the "
           "optimal monotone recalibration on test-day data — remaps the raw "
           "scores through it, and bids with the result in the same paired "
           "lognormal-market simulation (market prices and contribution "
           "machinery reused from Tier-B section E; paired block bootstrap, "
           "200 blocks, 10,000 resamples).",
           "",
           f"Sanity: recomputing the runs' raw/Platt/isotonic profits from "
           f"the saved predictions reproduces each run's bidding CSV with "
           f"max abs error {max_repro_err:.4f} currency units.",
           "",
           md_table(df[["origin", "model", "gap", "n", "profit_raw",
                        "profit_isotonic_oracle", "delta_profit",
                        "delta_per_1000", "ci_lo", "ci_hi", "p_boot",
                        "ci_excludes_0"]]),
           "",
           f"- The isotonic test-day oracle beats raw in {n_beats} of 24 "
           f"cells on point estimate; {n_sig_pos} cells are significantly "
           f"positive and {n_sig_neg} significantly negative (95% CI "
           f"excludes 0).",
           f"- Median profit delta vs raw = {med_per1000:+.4f} per 1000 "
           f"impressions (median total delta {med_delta:+.1f} currency "
           f"units).",
           "- Contrast with the Tier-B binned oracle: median −0.3698 per "
           "1000 impressions, significantly negative in 23 of 24 cells "
           "(results/tierb/rq4_bootstrap.csv).",
           ""]
    with open(md_path, "a") as f:
        f.write("\n".join(sec))
    print("appended WS5 section to results/tierc_analysis.md")
else:
    print("WS5 section already present; skipped append")
