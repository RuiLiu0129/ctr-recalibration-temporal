#!/usr/bin/env python3
"""One-command reproduction of the paper's quantitative claims (v9) from the
unrounded CSVs in this directory. Requires pandas and scipy. Run from this
directory:   python3 reproduce_paper_numbers.py

Each check prints the recomputed value next to the value printed in the paper
and marks it OK / MISMATCH. It also writes two derived tables that the paper
summarises but does not print in full:
  sweep_fixed_cells_three_seeds.csv   per seed x origin x calibrator x slice length
  sweep_harm_by_seed_origin.csv       harm counts per seed x origin x slice length
Exit code is the number of mismatches.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

bad = 0
def check(label, got, want, tol=0.0):
    global bad
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) else (got == want)
    bad += (not ok)
    print(f"[{'OK' if ok else 'MISMATCH'}] {label}: recomputed {got} | paper {want}")

# ---------------------------------------------------------------- 6-h grid
g = pd.read_csv("grid48_unrounded.csv")
g["harm"] = g.delta_ece > 0
check("harmful applications, seed 42, 6 h (of 48)", int(g.harm.sum()), 16)
rel = g.delta_ece / g.raw_ece
check("largest relative ECE reduction (%)", round(-rel.min() * 100), 61)
check("largest relative ECE increase (%)", round(rel.max() * 100), 107)
check("max |calibrated AUC - raw AUC|", round((g.cal_auc - g.raw_auc).abs().max(), 4), 0.0002)
dll = g.cal_logloss - g.raw_logloss
check("log-loss delta range", (round(dll.min(), 4), round(dll.max(), 4)), (-0.0009, 0.0006))
check("ECE and log loss move in opposite directions (of 48)", int(((g.delta_ece > 0) != (dll > 0)).sum()), 9)

h = pd.read_csv("seed_harm_counts.csv")
per_seed = h.groupby("seed").harm_cells.sum().to_dict()
check("harmful applications per seed (42/43/44)", tuple(per_seed[s] for s in (42, 43, 44)), (16, 14, 17))

# ------------------------------------------------- slice-vs-target bias rules
b = pd.read_csv("slice_target_bias.csv").merge(g, on=["origin", "model", "gap"])
sign_rule = np.sign(b.slice_signed_bias) != np.sign(b.target_signed_bias_raw)
over_rule = (b.target_signed_bias_raw - b.slice_signed_bias).abs() > b.target_signed_bias_raw.abs()
check("sign-flip rule correct (of 48)", int((sign_rule == b.harm).sum()), 40)
check("overshoot rule correct (of 48)", int((over_rule == b.harm).sum()), 42)
check("overshoot rule: flagged applications all harmed", (int(over_rule.sum()), int((over_rule & b.harm).sum())), (10, 10))

# ---------------------------------------------------- three-seed sweep (Table V)
S = pd.read_csv("sweep_seeds_cells.csv")
p = S.pivot_table(index=["seed", "calib_hours", "origin", "model", "bucket"], columns="variant", values="ece").reset_index()
parts = []
for c in ("platt", "isotonic"):
    t = p[["seed", "calib_hours", "origin", "model", "bucket"]].copy(); t["calibrator"] = c; t["delta_ece"] = p[c] - p["raw"]; parts.append(t)
D = pd.concat(parts); D["gap"] = D.bucket.str[-1].astype(int)
S42 = pd.read_csv("sweep_seed42_delta_ece.csv").rename(columns={"variant": "calibrator"}); S42["seed"] = 42
six = []
for o, f in ((1, pd.read_csv("seed_o1_delta_flags.csv")), (2, pd.read_csv("seed_o2_delta_ece.csv"))):
    for s in (43, 44):
        t = f[["model", "gap", "calibrator"]].copy(); t["delta_ece"] = f[f"dECE_s{s}"]; t["origin"] = o; t["seed"] = s; t["calib_hours"] = 6; six.append(t)
cols = ["seed", "calib_hours", "origin", "model", "gap", "calibrator", "delta_ece"]
A = pd.concat([D[cols], S42[cols], pd.concat(six)[cols]])
F = A[A.gap >= 1].copy(); F["harm"] = F.delta_ece > 0
F.sort_values(cols[:-1]).to_csv("sweep_fixed_cells_three_seeds.csv", index=False)
HSO = F.groupby(["seed", "origin", "calib_hours"]).harm.sum().unstack()
HSO.to_csv("sweep_harm_by_seed_origin.csv")
print("\nHarms per seed x origin (fixed cells, of 18):\n" + HSO.to_string())
hs = F.groupby(["seed", "calib_hours"]).harm.sum().unstack()
for hrs, want in ((1, (15, 15, 18)), (3, (16, 16, 20)), (6, (13, 14, 17)), (12, (4, 5, 3)), (24, (9, 11, 11))):
    check(f"Table V harms/seed at {hrs} h", tuple(int(hs.loc[s, hrs]) for s in (42, 43, 44)), want)
lr = F[F.model == "lr"].groupby(["seed", "calib_hours"]).harm.sum().unstack()
check("Table V LR harms/seed at 6 h", tuple(int(lr.loc[s, 6]) for s in (42, 43, 44)), (8, 8, 8))
check("Table V LR harms/seed at 12 h", tuple(int(lr.loc[s, 12]) for s in (42, 43, 44)), (2, 1, 1))
check("12 h has the fewest harms in every seed x origin", bool((HSO.idxmin(axis=1) == 12).all()), True)
mean_min = F.groupby(["seed", "origin", "calib_hours"]).delta_ece.mean().unstack().idxmin(axis=1)
check("seed x origin combinations where 12 h also has the lowest mean dECE", int((mean_min == 12).sum()), 5)
tv = (F.groupby(["origin", "calibrator", "calib_hours"]).delta_ece.mean().unstack() * 1e4).round(0)
want = {(1, "isotonic"): [-5, -13, -11, -38, -31], (1, "platt"): [-4, -2, -7, -17, -11],
        (2, "isotonic"): [18, 7, 2, -30, -30], (2, "platt"): [0, 6, 3, -14, -9]}
for k, v in want.items():
    check(f"Table V mean dECE x1e-4, origin {k[0]} {k[1]}", [int(x) + 0 for x in tv.loc[k].tolist()], v)

# ------------------------------------------------- size-matched control
M = pd.read_csv("mechanism_cells.csv"); M["cond"] = M.condition.where(~M.condition.str.startswith("sub"), "sub")
cell = M.groupby(["origin", "model", "gap", "calibrator", "cond"]).delta_ece.mean().unstack()
fx = cell[cell.index.get_level_values("gap") >= 1]
m = fx.mean()
check("control mean dECE full12 / hour1 / sub (gaps +1..+3)", tuple(round(m[c], 4) for c in ("full12", "hour1", "sub")), (-0.0024, -0.0001, -0.0021))
rec = (m["sub"] - m["hour1"]) / (m["full12"] - m["hour1"])
check("control recovery (gaps +1..+3)", round(rec, 2), 0.87)
ga = cell.mean(); rec_all = (ga["sub"] - ga["hour1"]) / (ga["full12"] - ga["hour1"])
check("control recovery incl. gap 0", round(rec_all, 2), 0.88, tol=0.005)
oc = fx.groupby(level=["origin", "calibrator"]).mean()
r = (oc["sub"] - oc["hour1"]) / (oc["full12"] - oc["hour1"])
check("control recovery range across origin x calibrator", (round(r.min(), 2), round(r.max(), 2)), (0.80, 0.99))
check("control harms full12 / hour1 / sub (of 36)", tuple(int((fx[c] > 0).sum()) for c in ("full12", "hour1", "sub")), (4, 15, 4))
Mt = M[M.gap >= 1]
ft = Mt.groupby(["cond", "calibrator"]).fit_time_s.median()
check("control same-run fit speed-up Platt / isotonic", (round(ft["full12", "platt"] / ft["sub", "platt"]), round(ft["full12", "isotonic"] / ft["sub", "isotonic"])), (20, 25))
check("control fit medians full12 Platt / iso (s)", (round(ft["full12", "platt"], 2), round(ft["full12", "isotonic"], 2)), (1.14, 0.58))
check("control fit medians sub Platt / iso (s)", (round(ft["sub", "platt"], 3), round(ft["sub", "isotonic"], 3)), (0.057, 0.023))

# ------------------------------------------------- timings (Table III)
T = pd.read_csv("sweep_seeds_timings.csv")
med = T.groupby(["calib_hours", "variant"]).fit_time_s.median()
for hrs, want in ((1, (0.085, 0.032)), (3, (0.26, 0.11)), (12, (1.41, 0.87)), (24, (2.80, 2.11))):
    nd = 3 if hrs == 1 else 2
    check(f"Table III fit medians at {hrs} h Platt / iso (s)", (round(med[hrs, "platt"], nd), round(med[hrs, "isotonic"], nd)), want)
ap = T.groupby("variant").apply_time_s.median()
check("apply-time medians within 0.2-0.5 s", bool(ap.between(0.2, 0.5).all()), True)

# ------------------------------------------------- RQ4
db = pd.read_csv("delta_ece_bootstrap.csv")
check("dECE bootstrap intervals excluding zero (of 48)", int(db.excludes_zero.sum()), 43)
rq = pd.read_csv("rq4_bootstrap.csv")
rec_ = rq[rq.comparison.isin(["platt-raw", "isotonic-raw"])]
check("profit intervals excluding zero (of 48)", int(rec_.ci_excludes_0.sum()), 24)
orc = pd.read_csv("rq4_binned_oracle.csv")
mm = rec_.assign(variant=rec_.comparison.str.replace("-raw", "", regex=False)).merge(
    orc[["origin", "model", "gap", "profit_raw"]], on=["origin", "model", "gap"]).merge(
    db[["origin", "model", "gap", "variant", "delta_ece"]], on=["origin", "model", "gap", "variant"])
mm["pct"] = mm.delta_profit / mm.profit_raw * 100
check("profit delta range (%)", (round(mm.pct.min(), 2), round(mm.pct.max(), 2)), (-0.21, 0.26))
check("sign agreement dECE vs dProfit (of 48)", int((mm.delta_ece * mm.pct < 0).sum()), 40)
for lab, sub, want in (("pooled", mm, -0.69), ("origin 1", mm[mm.origin == 1], -0.770), ("origin 2", mm[mm.origin == 2], -0.783)):
    nd = 2 if lab == "pooled" else 3
    check(f"Pearson r {lab}", round(pearsonr(sub.delta_ece, sub.pct)[0], nd), want)
co = rq[rq.comparison == "calib_oracle-raw"]
check("binned oracle below raw (point / significant)", (int((co.delta_profit < 0).sum()), int(((co.delta_profit < 0) & co.ci_excludes_0).sum())), (24, 23))
iso = pd.read_csv("rq4_isotonic_oracle.csv")
check("isotonic oracle median gain per 1,000", round(iso.delta_per_1000.median(), 3), 0.147)
lab = (orc.profit_label_oracle / orc.profit_raw)
check("label oracle multiple range", (round(lab.min(), 2), round(lab.max(), 2)), (1.75, 1.97))

print(f"\n{bad} mismatch(es)")
sys.exit(bad)
