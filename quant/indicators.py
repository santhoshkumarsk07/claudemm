"""Technical indicators (pandas, no TA library needed).

Every function is causal: the value at bar t uses only bars <= t. Wilder smoothing
(RSI, ATR, ADX) is an EMA with alpha = 1/n, which matches most charting platforms
once enough history has passed (the first ~3n bars are a warm-up period).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder(-d.clip(upper=0), n)
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn > 0, 100.0).where(up.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid, sd = sma(close, n), close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo,
                         "bb_pctb": (close - lo) / (up - lo), "bb_width": (up - lo) / mid})


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / tr
    minus_di = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": _wilder(dx, n)})


def vwap(df: pd.DataFrame, n: int | None = None) -> pd.Series:
    """Intraday: VWAP anchored to each session. Daily bars: rolling n-day VWAP
    (a session VWAP needs intraday data, which free sources only keep for weeks)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv, v = tp * df["volume"], df["volume"]
    if n is None:
        day = pd.Index(df.index.date)
        out = pv.groupby(day).cumsum() / v.groupby(day).cumsum().replace(0, np.nan)
    else:
        out = pv.rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum().replace(0, np.nan)
    return out.where(v.rolling(n or 1).sum() > 0)  # NaN when the feed has no volume (indices)


def add_all(df: pd.DataFrame, daily: bool = True) -> pd.DataFrame:
    """Every indicator the strategies, ML features and dashboard use."""
    out = df.copy()
    c = out["close"]
    for n in (20, 50, 200):
        out[f"sma{n}"] = sma(c, n)
    for n in (9, 21, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["rsi14"] = rsi(c)
    out = out.join(macd(c)).join(bollinger(c)).join(adx(out))
    out["atr14"] = atr(out)
    out["atr_pct"] = out["atr14"] / c * 100
    out["vwap"] = vwap(out, 20 if daily else None)
    out["vol_sma20"] = sma(out["volume"], 20)
    out["vol_ratio"] = (out["volume"] / out["vol_sma20"]).where(out["vol_sma20"] > 0)
    return out
