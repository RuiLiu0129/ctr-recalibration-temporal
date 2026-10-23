"""Simulated second-price bidding payoff.

Avazu contains no auction logs, so the market side is simulated: for each
impression the highest competing bid ("market price") is drawn once from
LogNormal(market_mu, market_sigma) in the same currency units as
``value_per_click`` (seeded, and *shared across strategies* so the comparison
is paired). Our bid is ``bid_shading * p * value_per_click``.

Rules per impression:
  - win if our bid > market price; pay the market price (second price);
  - realized value = value_per_click if the impression truly clicks else 0;
  - profit contribution = realized value - payment (0 if we lose).

Strategies compared per test bucket:
  - "raw"        : the model's raw probabilities
  - "platt" / "isotonic": probabilities recalibrated on the recent slice
  - "oracle"     : unattainable upper bound that bids value_per_click exactly
                   on impressions that will click and 0 on the rest
                   (uses ground-truth labels; never deployable)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def draw_market_prices(n: int, cfg: dict, seed: int) -> np.ndarray:
    """One market price per impression, shared by all strategies."""
    bid_cfg = cfg["bidding"]
    rng = np.random.default_rng(seed)
    return rng.lognormal(
        mean=float(bid_cfg["market_mu"]),
        sigma=float(bid_cfg["market_sigma"]),
        size=n,
    )


def simulate_strategy(
    name: str,
    y: np.ndarray,
    p: np.ndarray,
    market_price: np.ndarray,
    cfg: dict,
) -> dict:
    """Run one bidding strategy over a bucket and return aggregate results."""
    bid_cfg = cfg["bidding"]
    value = float(bid_cfg["value_per_click"])
    bids = float(bid_cfg.get("bid_shading", 1.0)) * np.asarray(p, dtype=np.float64) * value
    won = bids > market_price
    clicks_won = won & (np.asarray(y) == 1)
    total_value = float(value * clicks_won.sum())
    total_cost = float(market_price[won].sum())
    return {
        "strategy": name,
        "impressions": int(len(y)),
        "won": int(won.sum()),
        "win_rate": float(won.mean()) if len(y) else float("nan"),
        "clicks_won": int(clicks_won.sum()),
        "total_value": total_value,
        "total_cost": total_cost,
        "profit": total_value - total_cost,
    }


def oracle_probs(y: np.ndarray) -> np.ndarray:
    """Ground-truth 'probabilities' for the oracle strategy (upper bound)."""
    return np.asarray(y, dtype=np.float64)


def run_bidding_sim(
    y: np.ndarray,
    prob_variants: dict[str, np.ndarray],
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    """Compare all strategies on one bucket.

    ``prob_variants`` maps variant name -> probabilities (e.g. raw / platt /
    isotonic); the oracle row is added automatically.
    """
    market_price = draw_market_prices(len(y), cfg, seed)
    rows = [simulate_strategy(name, y, p, market_price, cfg) for name, p in prob_variants.items()]
    rows.append(simulate_strategy("oracle", y, oracle_probs(y), market_price, cfg))
    return pd.DataFrame(rows)
