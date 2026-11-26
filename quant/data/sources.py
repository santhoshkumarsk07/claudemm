"""Free market-data sources.

YahooSource  - yfinance (unofficial Yahoo Finance API). Free, no key, personal use.
CSVSource    - any OHLCV CSV you downloaded legally (e.g. a Kaggle dataset).

Known limits of the free sources (also shown in the dashboard):
- Yahoo data is unofficial and can have gaps, late corrections or a missing last bar.
- Intraday history is short: 1m bars for ~7 days (max 30 days back), 2m-30m for 60 days, 60m for 730 days.
- Live quotes are delayed (typically up to 15 minutes for NSE) and are not exchange-licensed.
- Index tickers (^NSEI, ^NSEBANK) report zero volume, so volume filters are skipped for them.
- No free historical options chains or tick data. No point-in-time index constituents.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..clock import IST

log = logging.getLogger("quant")
COLS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataSourceError(RuntimeError):
    pass


def _normalise(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):  # yf.download returns (field, ticker) columns
        df = df.droplevel(1, axis=1) if df.columns.nlevels > 1 else df
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    if "adj_close" not in df:
        df["adj_close"] = df["close"]
    df = df[COLS].copy()
    idx = pd.DatetimeIndex(df.index)
    if interval == "1d":  # daily bars are dates, not instants
        idx = (idx.tz_convert(IST) if idx.tz is not None else idx).tz_localize(None).normalize()
    else:
        idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    df.index = idx
    df.index.name = "ts"
    return df.astype(float)


class YahooSource:
    name = "yahoo"
    terms = "Unofficial Yahoo Finance data via yfinance: personal, research and educational use only."
    MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "1h": 730, "1d": None}

    def __init__(self, min_interval_seconds: float = 1.0, retries: int = 4):
        self.min_interval = min_interval_seconds
        self.retries = retries
        self._last_call = 0.0

    def _throttle(self) -> None:  # be polite: at most one request per second
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def fetch(self, symbol: str, interval: str = "1d", start: str | None = None) -> pd.DataFrame:
        import yfinance as yf

        if interval not in self.MAX_DAYS:
            raise DataSourceError(f"Unsupported interval {interval}. Use one of {list(self.MAX_DAYS)}.")
        if interval == "1d":
            kw = {"start": start} if start else {"period": "max"}
        else:  # free intraday history is limited; ask for the most Yahoo allows
            kw = {"period": f"{self.MAX_DAYS[interval]}d"}
        last_err = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                raw = yf.download(symbol, interval=interval, auto_adjust=False, progress=False, threads=False, **kw)
                if raw is not None and len(raw):
                    df = _normalise(raw, interval).dropna(subset=["close"])
                    if len(df):
                        return df
                last_err = "no rows returned (bad symbol, market holiday, or Yahoo is throttling)"
            except Exception as e:  # network errors, YFRateLimitError, JSON errors
                last_err = f"{type(e).__name__}: {e}"
            delay = min(2 ** attempt * 2, 30)
            log.warning("Yahoo %s %s attempt %d failed (%s); retrying in %ds", symbol, interval, attempt + 1, last_err, delay)
            time.sleep(delay)
        raise DataSourceError(f"Yahoo gave no data for {symbol} ({interval}): {last_err}")


class CSVSource:
    """Import a legally obtained OHLCV CSV (Kaggle, NSE bhavcopy you saved, etc.)."""

    name = "csv"
    ALIASES = {"date": "ts", "datetime": "ts", "timestamp": "ts", "time": "ts", "adj close": "adj_close",
               "adjclose": "adj_close", "adj_close": "adj_close", "open": "open", "high": "high", "low": "low",
               "close": "close", "volume": "volume", "vol": "volume"}

    def fetch(self, path: str | Path, interval: str = "1d") -> pd.DataFrame:
        df = pd.read_csv(path)
        df = df.rename(columns={c: self.ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns})
        missing = {"ts", "open", "high", "low", "close"} - set(df.columns)
        if missing:
            raise DataSourceError(f"{path}: missing columns {sorted(missing)}")
        if "volume" not in df:
            df["volume"] = 0.0
        df = df.set_index(pd.to_datetime(df.pop("ts"), dayfirst=False))
        return _normalise(df, interval).dropna(subset=["close"])


def data_kind(interval: str, last_ts, now: datetime | None = None) -> str:
    """HISTORICAL (end-of-day), DELAYED (recent intraday from a free feed) or HISTORICAL INTRADAY."""
    if interval == "1d":
        return "HISTORICAL (end-of-day)"
    now = now or datetime.now(IST)
    age_min = (now - pd.Timestamp(last_ts).to_pydatetime()).total_seconds() / 60
    return "DELAYED (free feed, ~15 min or more)" if age_min < 120 else "HISTORICAL (intraday)"
