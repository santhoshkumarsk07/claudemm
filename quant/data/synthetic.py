"""Synthetic prices for unit tests only. Labelled SYNTHETIC and refused by the market-data store."""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE = "SYNTHETIC (test only, not market data)"


def daily_bars(n: int = 1500, seed: int = 1, start: str = "2015-01-01", drift: float = 0.0003,
               vol: float = 0.012, regime_len: int = 120, base: float = 1000.0) -> pd.DataFrame:
    """Geometric random walk with trending regimes, so strategies have something to find."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    regimes = np.repeat(rng.normal(0, 0.0012, n // regime_len + 1), regime_len)[:n]
    r = drift + regimes + rng.normal(0, vol, n)
    close = base * np.exp(np.cumsum(r))
    gap = rng.normal(0, vol / 3, n)
    open_ = np.r_[base, close[:-1]] * np.exp(gap)
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.lognormal(13, 0.4, n) * (1 + 5 * np.abs(r))
    df = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "adj_close": close, "volume": volume}, index=idx)
    df.index.name = "ts"
    df.attrs["sources"] = [SOURCE]
    return df
