# Tier-F Experiment A — bidding-sim parameter sweep

48 RQ4 deltas (2 origins x 3 models x 4 gaps x {platt, isotonic}, seed 42) per (mu, sigma) cell; delta_pct = (profit_recal - profit_raw) / profit_raw * 100.

Centre cell (mu=-2.2, sigma=0.9) reproduction vs the paper's supplement (rq4_bootstrap.csv): max abs deviation in delta_profit = 0.000000 currency units; max abs deviation in delta_pct = 2.53e-14 pp. Paper range check: [-0.207%, 0.263%] (paper: [-0.207%, +0.263%]).

| mu | sigma | min delta_pct | max delta_pct | mean | n positive (of 48) |
|---|---|---|---|---|---|
| -2.6 | 0.6 | -0.268 | 0.288 | 0.0317 | 33 |
| -2.6 | 0.9 | -0.157 | 0.283 | 0.0340 | 34 |
| -2.6 | 1.2 | -0.127 | 0.313 | 0.0306 | 31 |
| -2.2 | 0.6 | -0.296 | 0.343 | 0.0395 | 34 |
| -2.2 | 0.9 | -0.207 | 0.263 | 0.0291 | 36 |
| -2.2 | 1.2 | -0.157 | 0.285 | 0.0366 | 34 |
| -1.8 | 0.6 | -0.357 | 0.306 | 0.0369 | 30 |
| -1.8 | 0.9 | -0.320 | 0.292 | 0.0192 | 31 |
| -1.8 | 1.2 | -0.273 | 0.273 | 0.0196 | 33 |

Total wall time: 67s. Saved predictions only; no retraining.
