"""Feature engineering and labels.

Features at row t use bars <= t only (they are known at the close of t).
Labels describe what a trade decided at that close would have earned: enter at the
NEXT open (t+1) and exit `horizon` sessions later at the open (t+1+h). So the label of
row t is only known at t+1+h; walk-forward training purges rows whose label is not yet
known at the time of each prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import strategies

FEATURES = [
    "ret_1", "ret_5", "ret_10", "ret_20", "vol_20", "rsi", "macd_hist_n", "dist_ema21", "dist_ema50",
    "dist_ema200", "ema21_50", "adx", "di_diff", "atr_pct", "bb_pctb", "bb_width", "vol_ratio", "has_volume",
    "dist_high_252", "dist_low_252", "algo_setup",
]
MARKET_FEATURES = ["mkt_ret_5", "mkt_ret_20", "mkt_dist_ema200", "mkt_adx"]


def build(d: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """d: indicator frame from indicators.add_all. market: the NIFTY indicator frame (optional context)."""
    c = d["close"]
    r = c.pct_change()
    f = pd.DataFrame(index=d.index)
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_20"] = r.rolling(20).std()
    f["rsi"] = d["rsi14"] / 100
    f["macd_hist_n"] = d["macd_hist"] / c
    for n in (21, 50, 200):
        f[f"dist_ema{n}"] = c / d[f"ema{n}"] - 1
    f["ema21_50"] = d["ema21"] / d["ema50"] - 1
    f["adx"] = d["adx"] / 100
    f["di_diff"] = (d["plus_di"] - d["minus_di"]) / 100
    f["atr_pct"] = d["atr_pct"] / 100
    f["bb_pctb"] = d["bb_pctb"]
    f["bb_width"] = d["bb_width"]
    f["has_volume"] = d["vol_ratio"].notna().astype(float)
    f["vol_ratio"] = d["vol_ratio"].fillna(1.0).clip(0, 10)
    f["dist_high_252"] = c / c.rolling(252, min_periods=60).max() - 1
    f["dist_low_252"] = c / c.rolling(252, min_periods=60).min() - 1
    f["algo_setup"] = strategies.algo_setup(d).astype(float)  # the hybrid model sees the rule state
    if market is not None:
        mc = market["close"].reindex(d.index)  # only same-day or earlier market closes are joined
        f["mkt_ret_5"] = mc.pct_change(5, fill_method=None)
        f["mkt_ret_20"] = mc.pct_change(20, fill_method=None)
        f["mkt_dist_ema200"] = mc / market["ema200"].reindex(d.index) - 1
        f["mkt_adx"] = market["adx"].reindex(d.index) / 100
    return f.replace([np.inf, -np.inf], np.nan)


def forward_return(d: pd.DataFrame, horizon: int) -> pd.Series:
    """Return of buying at the next open and selling `horizon` sessions after that, at the open."""
    o = d["open"]
    return o.shift(-(horizon + 1)) / o.shift(-1) - 1


def labels(d: pd.DataFrame, horizon: int, cost: float = 0.003) -> tuple[pd.Series, pd.Series]:
    """(classification label, regression target). Up = return beats a ~0.3% round-trip cost."""
    fwd = forward_return(d, horizon)
    y = (fwd > cost).astype(float).where(fwd.notna())
    return y, fwd
