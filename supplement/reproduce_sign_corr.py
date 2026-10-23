#!/usr/bin/env python3
"""Reproduce the Sec. IV-E (RQ4) sign-agreement count and Pearson correlations
from the unrounded CSVs in this supplement.

Definitions (exactly as used for the paper's numbers):
  * dECE    = calibrated ECE - raw ECE (delta_ece_bootstrap.csv, full precision).
  * dProfit = recalibrated profit - raw profit, expressed as a PERCENTAGE of raw
              profit, i.e. the quantity printed in Table V of the paper
              (rq4_bootstrap.csv delta_profit / rq4_binned_oracle.csv profit_raw * 100).
  * AGREEMENT: a cell agrees iff sign(dECE) and sign(dProfit) are OPPOSITE
    (calibration improves, dECE < 0, while profit rises, dProfit > 0; or
    calibration worsens while profit falls).
  * Zero handling: a delta of exactly 0.0 has no sign and would be counted as
    NON-agreement. In the unrounded data NO delta_ece and NO profit delta is
    exactly zero (asserted below), so this rule never binds; it matters only
    for the rounded PDF tables, where two deltas print as +0.00%.

Expected output (paper, rounded): 40 of 48 agreements;
  r = -0.69 pooled, -0.77 at origin 1, -0.78 at origin 2
  (unrounded: -0.6866, -0.7695, -0.7830).

Run:  python3 reproduce_sign_corr.py   (requires pandas, scipy; reads only
files in this directory)
"""
import pandas as pd
from scipy.stats import pearsonr

boot = pd.read_csv("delta_ece_bootstrap.csv")          # dECE per cell
rq4 = pd.read_csv("rq4_bootstrap.csv")                 # absolute profit deltas
rq4["variant"] = rq4["comparison"].str.replace("-raw", "", regex=False)
orc = pd.read_csv("rq4_binned_oracle.csv")             # raw profit per cell

m = boot.merge(
    rq4[["origin", "model", "gap", "variant", "delta_profit"]],
    on=["origin", "model", "gap", "variant"],
).merge(orc[["origin", "model", "gap", "profit_raw"]], on=["origin", "model", "gap"])
assert len(m) == 48, len(m)

m["dprofit_pct"] = m["delta_profit"] / m["profit_raw"] * 100.0

assert (m["delta_ece"] != 0).all(), "an exact-zero dECE occurs; see docstring"
assert (m["delta_profit"] != 0).all(), "an exact-zero dProfit occurs; see docstring"

m["agree"] = (m["delta_ece"] * m["dprofit_pct"]) < 0
print(f"sign agreements: {int(m['agree'].sum())} of 48")

for label, sub in [("pooled", m), ("origin 1", m[m.origin == 1]), ("origin 2", m[m.origin == 2])]:
    r = pearsonr(sub["delta_ece"], sub["dprofit_pct"])[0]
    print(f"Pearson r ({label}): {r:.4f}")
