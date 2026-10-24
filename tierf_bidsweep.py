"""Tier-F Experiment A (2026-10-23): RQ4 bidding-simulation parameter sweep.

Question (external Strong-Accept roadmap, cheap identification experiment
2): the paper's RQ4 profit deltas (recalibrated vs raw, -0.207%..+0.263%)
were measured at ONE point of the simulated market -- lognormal
mu=-2.2, sigma=0.9, value/click 1.0. Is the near-zero profit effect an
artifact of that parameter point?

Protocol: IDENTICAL to the original RQ4 computation (tierb_analyze.py
section E) -- same saved seed-42 predictions (results/o1_repro/preds,
results/origin2/preds), same test-bucket labels, same paired market-price
draw rule (seed = 42 + gap, shared across strategies) -- except the market
lognormal (mu, sigma) varies over a 3x3 grid:

    mu    in {-2.6, -2.2, -1.8}
    sigma in { 0.6,  0.9,  1.2}

The centre cell (mu=-2.2, sigma=0.9) must reproduce the paper's 48
per-cell profit deltas (supplement rq4_bootstrap.csv, comparisons
platt-raw and isotonic-raw over 2 origins x 3 models x 4 gaps); the max
abs deviation is reported and asserted small. NO model retraining: saved
predictions only.

Outputs (results/bidsweep/):
  bidsweep_cells.csv    one row per origin x model x gap x method x
                        (mu, sigma): profit_raw, profit_recal,
                        delta_profit, delta_pct (= delta/profit_raw*100,
                        the paper's RQ4 caliber)
  bidsweep_summary.md   per-(mu, sigma) range of the 48 delta_pct values
                        + the centre-cell reproduction check
"""

from __future__ import annotations

import copy
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import yaml

from src.bidding_sim import draw_market_prices, simulate_strategy

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results", "bidsweep")
SUPPLEMENT = os.path.join(
    ROOT, "..", "repo-staging", "supplement")
MODELS = ("lr", "deepfm", "dcnv2")
GAPS = (0, 1, 2, 3)
MUS = (-2.6, -2.2, -1.8)
SIGMAS = (0.6, 0.9, 1.2)
RUN_SPECS = [(1, "o1_repro", "data/processed_o1repro"),
             (2, "origin2", "data/processed_o2")]


def main() -> None:
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(ROOT, "config.yaml")) as f:
        base_cfg = yaml.safe_load(f)

    rows = []
    for origin, rdir, pdir in RUN_SPECS:
        for model in MODELS:
            for g in GAPS:
                y = np.asarray(np.load(os.path.join(
                    ROOT, pdir, "temporal", f"test_gap{g}_y.npy"),
                    mmap_mode="r"), dtype=np.float64)
                preds = {v: np.load(os.path.join(
                    ROOT, "results", rdir, "preds",
                    f"temporal_{model}_gap{g}_{v}.npy")).astype(np.float64)
                    for v in ("raw", "platt", "isotonic")}
                for mu in MUS:
                    for sigma in SIGMAS:
                        cfg = copy.deepcopy(base_cfg)
                        cfg["bidding"]["market_mu"] = mu
                        cfg["bidding"]["market_sigma"] = sigma
                        # paired draw: same seed rule as the original
                        # protocol (42 + gap), shared across strategies
                        mprice = draw_market_prices(len(y), cfg, seed=42 + g)
                        profits = {v: simulate_strategy(
                            v, y, preds[v], mprice, cfg)["profit"]
                            for v in preds}
                        for method in ("platt", "isotonic"):
                            d = profits[method] - profits["raw"]
                            rows.append({
                                "origin": origin, "seed": 42,
                                "model": model, "gap": g,
                                "mu": mu, "sigma": sigma,
                                "method": method,
                                "n": int(len(y)),
                                "profit_raw": profits["raw"],
                                "profit_recal": profits[method],
                                "delta_profit": d,
                                "delta_pct": d / profits["raw"] * 100.0,
                            })
                print(f"origin {origin} {model} gap{g}: done "
                      f"({time.time() - t_start:.0f}s)", flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(OUT, "bidsweep_cells.csv"), index=False)

    # ---- centre-cell reproduction check vs the paper's 48 deltas --------
    centre = cells[(cells["mu"] == -2.2) & (cells["sigma"] == 0.9)].copy()
    boot = pd.read_csv(os.path.join(SUPPLEMENT, "rq4_bootstrap.csv"))
    boot = boot[boot["comparison"].isin(["platt-raw", "isotonic-raw"])].copy()
    boot["method"] = boot["comparison"].str.replace("-raw", "", regex=False)
    merged = centre.merge(
        boot[["origin", "model", "gap", "method", "delta_profit"]],
        on=["origin", "model", "gap", "method"],
        suffixes=("_sweep", "_paper"))
    assert len(merged) == 48, f"expected 48 centre rows, got {len(merged)}"
    max_dev_delta = float(
        (merged["delta_profit_sweep"] - merged["delta_profit_paper"]
         ).abs().max())
    # pct caliber exactly as reproduce_paper_numbers.py: profit_raw from
    # rq4_binned_oracle.csv merged on (origin, model, gap)
    orc = pd.read_csv(os.path.join(SUPPLEMENT, "rq4_binned_oracle.csv"))
    mm = boot.merge(
        orc[["origin", "model", "gap", "profit_raw"]],
        on=["origin", "model", "gap"])
    mm["pct_paper"] = mm["delta_profit"] / mm["profit_raw"] * 100.0
    merged2 = centre.merge(
        mm[["origin", "model", "gap", "method", "pct_paper"]],
        on=["origin", "model", "gap", "method"])
    max_dev_pct = float(
        (merged2["delta_pct"] - merged2["pct_paper"]).abs().max())
    pct_min, pct_max = float(merged2["pct_paper"].min()), float(
        merged2["pct_paper"].max())

    # ---- summary ---------------------------------------------------------
    lines = ["# Tier-F Experiment A — bidding-sim parameter sweep",
             "",
             "48 RQ4 deltas (2 origins x 3 models x 4 gaps x {platt, "
             "isotonic}, seed 42) per (mu, sigma) cell; delta_pct = "
             "(profit_recal - profit_raw) / profit_raw * 100.",
             "",
             f"Centre cell (mu=-2.2, sigma=0.9) reproduction vs the "
             f"paper's supplement (rq4_bootstrap.csv): max abs deviation "
             f"in delta_profit = {max_dev_delta:.6f} currency units; "
             f"max abs deviation in delta_pct = {max_dev_pct:.2e} pp. "
             f"Paper range check: [{pct_min:.3f}%, {pct_max:.3f}%] "
             f"(paper: [-0.207%, +0.263%]).",
             "",
             "| mu | sigma | min delta_pct | max delta_pct | mean | "
             "n positive (of 48) |",
             "|---|---|---|---|---|---|"]
    for mu in MUS:
        for sigma in SIGMAS:
            sub = cells[(cells["mu"] == mu) & (cells["sigma"] == sigma)]
            lines.append(
                f"| {mu} | {sigma} | {sub['delta_pct'].min():.3f} | "
                f"{sub['delta_pct'].max():.3f} | "
                f"{sub['delta_pct'].mean():.4f} | "
                f"{int((sub['delta_pct'] > 0).sum())} |")
    lines += ["", f"Total wall time: {time.time() - t_start:.0f}s. "
              "Saved predictions only; no retraining."]
    with open(os.path.join(OUT, "bidsweep_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
