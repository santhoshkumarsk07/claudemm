"""Cleaning and data-quality checks. Bad bars are dropped and reported, never invented."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..clock import IST, expected_last_session
from .sources import data_kind


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (clean OHLCV, list of issues found). No forward-filling of prices."""
    issues = []
    n0 = len(df)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < n0:
        issues.append(f"{n0 - len(df)} duplicate timestamps removed")
    bad = df[["open", "high", "low", "close"]].isna().any(axis=1) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        issues.append(f"{int(bad.sum())} bars with missing or non-positive prices removed")
        df = df[~bad]
    # Yahoo sometimes reports a high below the close (or low above it). Repair the range, keep the bar.
    hi = df[["open", "close"]].max(axis=1)
    lo = df[["open", "close"]].min(axis=1)
    broken = (df["high"] < hi) | (df["low"] > lo)
    if broken.any():
        issues.append(f"{int(broken.sum())} bars had high/low inside the open-close range; range widened")
        df = df.assign(high=np.maximum(df["high"], hi), low=np.minimum(df["low"], lo))
    df = df.assign(volume=df["volume"].fillna(0).clip(lower=0))
    bad_print = _spike_and_revert(df["close"].to_numpy())
    if bad_print:
        days = ", ".join(df.index[i].strftime("%Y-%m-%d") for i in bad_print[:6])
        issues.append(f"{len(bad_print)} bad prints removed (price jumped >25% and reverted within 3 bars): {days}")
        df = df.drop(df.index[bad_print])
    return df, issues


def _spike_and_revert(close, jump: float = 0.25, back: float = 0.07, max_len: int = 3) -> list[int]:
    """Indices of bars that jump far from the previous close and return to it within `max_len` bars.
    Real splits and crashes do not snap back like this; feed errors (e.g. a 1/10-scale price) do."""
    out, i, n = [], 1, len(close)
    while i < n:
        ref = close[i - 1]
        if abs(close[i] / ref - 1) > jump:
            for k in range(1, max_len + 1):
                if i + k < n and abs(close[i + k] / ref - 1) <= back:
                    out += list(range(i, i + k))
                    i += k
                    break
        i += 1
    return out


def adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLC for splits and dividends using Yahoo's adj_close ratio.
    Backtests need this, or a 1:2 split looks like a 50% crash."""
    f = (df["adj_close"] / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    out = df.copy()
    for c in ("open", "high", "low", "close"):
        out[c] = df[c] * f
    out["volume"] = df["volume"] / f.where(f > 0, 1.0)
    return out


def report(df: pd.DataFrame, symbol: str, interval: str = "1d", now: datetime | None = None,
           stale_after_hours: float = 30, holidays: set[str] = frozenset()) -> dict:
    """Quality summary plus warnings shown next to every chart in the dashboard."""
    now = now or datetime.now(IST)
    if df is None or df.empty:
        return {"symbol": symbol, "rows": 0, "status": "MISSING", "warnings": [f"No {interval} data stored for {symbol}."]}
    attrs = dict(df.attrs)
    df, cleaned = clean(df)  # report on the data the strategies actually use
    df.attrs.update(attrs)
    warn = [f"Cleaning: {c}" for c in cleaned]
    last = df.index[-1]
    rets = df["close"].pct_change().abs()
    jumps = rets[rets > 0.2]
    if len(jumps):
        warn.append(f"{len(jumps)} daily moves above 20% (real news moves, or an unadjusted split): {', '.join(d.strftime('%Y-%m-%d') for d in jumps.index[:3])}")
    zero_vol = float((df["volume"] <= 0).mean() * 100)
    if zero_vol > 50:
        warn.append("Volume is missing (normal for Yahoo index tickers); volume filters are skipped.")
    elif zero_vol > 2:
        warn.append(f"{zero_vol:.1f}% of bars have zero volume.")
    status = "OK"
    if interval == "1d":
        bdays = pd.bdate_range(df.index[0], df.index[-1])
        missing = len(bdays.difference(df.index))
        gap_pct = missing / max(len(bdays), 1) * 100
        if gap_pct > 6:  # NSE has ~12-16 holidays a year, about 5-6% of weekdays
            warn.append(f"{missing} weekdays without a bar ({gap_pct:.1f}%), more than holidays explain.")
        exp = expected_last_session(now, holidays)
        age_h = (now.replace(tzinfo=None) - pd.Timestamp(last).to_pydatetime().replace(hour=15, minute=30)).total_seconds() / 3600
        if last.date() < exp:
            status = "STALE"
            warn.append(f"Newest bar is {last.date()}, expected {exp} (stale, or {exp} was a market holiday). "
                        "Run `python -m quant fetch`.")
        elif age_h > stale_after_hours + 72:
            status = "STALE"
    else:
        missing = None
        age_h = (now - pd.Timestamp(last).to_pydatetime()).total_seconds() / 3600
        warn.append("Intraday data from free feeds is delayed and limited to recent weeks.")
    return {
        "symbol": symbol, "interval": interval, "rows": len(df), "first": str(df.index[0])[:19], "last": str(last)[:19],
        "missing_weekdays": missing, "zero_volume_pct": round(zero_vol, 1), "age_hours": round(age_h, 1),
        "kind": data_kind(interval, last, now), "status": status, "warnings": warn,
        "sources": df.attrs.get("sources", []), "fetched_at": df.attrs.get("fetched_at"),
    }
