"""Trading strategies. Each returns boolean `entry` and `exit` Series computed on the close of
each bar from data up to that bar only. The simulator fills them at the next open.

Traditional:  ma_crossover, rsi_reversion, macd_trend, trend_following, algo (composite)
ML:           ml           (walk-forward probability only)
Hybrid:       hybrid       (algo setup AND the ML model agrees)
"""
from __future__ import annotations

import pandas as pd


def _cross_up(a: pd.Series, b) -> pd.Series:
    return (a > b) & (a.shift() <= (b.shift() if isinstance(b, pd.Series) else b))


def _cross_dn(a: pd.Series, b) -> pd.Series:
    return (a < b) & (a.shift() >= (b.shift() if isinstance(b, pd.Series) else b))


def volume_ok(d: pd.DataFrame) -> pd.Series:
    """Volume confirmation; passes automatically when the feed has no volume (index tickers)."""
    return (d["vol_ratio"] >= 1.0) | d["vol_ratio"].isna()


def volatility_ok(d: pd.DataFrame, pct: float = 0.85, window: int = 252) -> pd.Series:
    """Skip entries when ATR% is in the top 15% of the past year (panic regimes)."""
    cap = d["atr_pct"].rolling(window, min_periods=60).quantile(pct)
    return (d["atr_pct"] <= cap) | cap.isna()


def ma_crossover(d: pd.DataFrame, fast: str = "ema21", slow: str = "ema50") -> tuple[pd.Series, pd.Series]:
    return _cross_up(d[fast], d[slow]), _cross_dn(d[fast], d[slow])


def rsi_reversion(d: pd.DataFrame, low: float = 30, exit_at: float = 55) -> tuple[pd.Series, pd.Series]:
    """Buy an oversold bounce, but only in a long-term uptrend (close above SMA200)."""
    return _cross_up(d["rsi14"], low) & (d["close"] > d["sma200"]), d["rsi14"] >= exit_at


def macd_trend(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """MACD crossing its signal line, confirmed by price above the 200-day EMA."""
    return _cross_up(d["macd"], d["macd_signal"]) & (d["close"] > d["ema200"]), _cross_dn(d["macd"], d["macd_signal"])


def trend_following(d: pd.DataFrame, adx_min: float = 20) -> tuple[pd.Series, pd.Series]:
    up = (d["close"] > d["ema50"]) & (d["ema50"] > d["ema200"]) & (d["adx"] > adx_min) & (d["plus_di"] > d["minus_di"])
    return up & ~up.shift(fill_value=False), d["close"] < d["ema50"]


def algo_setup(d: pd.DataFrame) -> pd.Series:
    """The composite algorithmic long setup: trend + momentum + RSI band + volume + volatility filter."""
    return ((d["ema21"] > d["ema50"]) & (d["close"] > d["ema200"]) & (d["macd_hist"] > 0)
            & d["rsi14"].between(45, 70) & volume_ok(d) & volatility_ok(d))


def algo_exit(d: pd.DataFrame) -> pd.Series:
    return _cross_dn(d["ema21"], d["ema50"]) | ((d["close"] < d["ema50"]) & (d["macd_hist"] < 0))


def algo(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    s = algo_setup(d)
    return s & ~s.shift(fill_value=False), algo_exit(d)


def ml(d: pd.DataFrame, p_up: pd.Series, buy: float = 0.55, sell_margin: float = 0.1) -> tuple[pd.Series, pd.Series]:
    p = p_up.reindex(d.index)
    return p >= buy, p < buy - sell_margin


def hybrid(d: pd.DataFrame, p_up: pd.Series, threshold: float = 0.50, sell_margin: float = 0.1) -> tuple[pd.Series, pd.Series]:
    """Enter only while the algorithmic setup holds AND the model agrees; leave on either exit."""
    p = p_up.reindex(d.index)
    return algo_setup(d) & (p >= threshold), algo_exit(d) | (p < threshold - sell_margin)


TRADITIONAL = {"ma_crossover": ma_crossover, "rsi_reversion": rsi_reversion, "macd_trend": macd_trend,
               "trend_following": trend_following, "algo": algo}


def signals(name: str, d: pd.DataFrame, p_up: pd.Series | None = None, cfg: dict | None = None) -> tuple[pd.Series, pd.Series]:
    m = (cfg or {}).get("ml", {})
    if name == "ml":
        return ml(d, p_up, m.get("buy_threshold", 0.55))
    if name == "hybrid":
        return hybrid(d, p_up, m.get("hybrid_threshold", 0.50))
    entry, exit_ = TRADITIONAL[name](d)
    return entry.fillna(False).astype(bool), exit_.fillna(False).astype(bool)


def explain(name: str, d: pd.DataFrame, i: int = -1) -> str:
    """Plain-English reason for the latest signal state (shown in the dashboard and journal)."""
    r = d.iloc[i]
    parts = [f"close {r.close:.2f}", f"EMA21 {r.ema21:.2f} vs EMA50 {r.ema50:.2f}", f"EMA200 {r.ema200:.2f}",
             f"RSI {r.rsi14:.1f}", f"MACD hist {r.macd_hist:+.2f}", f"ADX {r.adx:.1f}", f"ATR {r.atr_pct:.2f}%"]
    if pd.notna(r.vol_ratio):
        parts.append(f"volume {r.vol_ratio:.2f}x avg")
    return f"{name}: " + ", ".join(parts)
