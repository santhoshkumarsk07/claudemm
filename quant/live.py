"""Live paper-trading session during NSE market hours (09:15-15:30 IST). Virtual money only.

What happens:
- Orders the agents queued last evening fill at the first 1-minute bar of the session
  (its open price, plus slippage and charges), exactly like the backtest's "next open" rule.
  If you start late, they fill at the latest price instead and the log says so.
- Every minute, each open position's stop-loss and take-profit are checked against the
  new 1-minute bars (stop first if a bar touches both; a gap fills at the bar's open).
- At 15:30 the account is marked to the last prices and the day is closed, so the evening
  agent run does not process this session a second time.
- New BUY/SELL decisions are still made after the close by the agents (`quant daily`),
  using the completed daily bar: the strategies and ML were tested on daily bars.

Data: Yahoo Finance 1-minute bars through yfinance. Free, unofficial and DELAYED (often
up to 15 minutes for NSE). The dashboard labels every live price as delayed.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Callable

import pandas as pd

from .clock import CLOSE, IST, OPEN, now_ist
from .paper import PaperAccount

log = logging.getLogger("quant")
LATE_FILL_MINUTES = 5


def yahoo_1m(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Today's 1-minute bars per symbol (IST index). Empty dict on failure; never raises."""
    import yfinance as yf

    from .data.sources import _normalise
    try:
        raw = yf.download(symbols, period="1d", interval="1m", group_by="ticker", auto_adjust=False,
                          progress=False, threads=True)
    except Exception as e:
        log.warning("Live quotes failed: %s", e)
        return {}
    out = {}
    for s in symbols:
        try:
            df = raw[s] if len(symbols) > 1 else raw
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(0, axis=1) if s in df.columns.get_level_values(0) else df.droplevel(1, axis=1)
            df = _normalise(df, "1m").dropna(subset=["close"])
            if len(df):
                out[s] = df
        except (KeyError, ValueError):
            continue
    return out


class LiveSession:
    def __init__(self, cfg: dict, account: PaperAccount, quotes: Callable[[list[str]], dict] = yahoo_1m,
                 clock: Callable[[], datetime] = now_ist, sleep: Callable[[float], None] = time.sleep,
                 poll_seconds: int = 60):
        self.cfg, self.acct = cfg, account
        self.quotes, self.clock, self.sleep, self.poll = quotes, clock, sleep, poll_seconds
        self.symbols = list(cfg["data"]["watchlist"])
        self.last_seen: dict[str, pd.Timestamp] = {}
        self.last_price: dict[str, float] = {}

    def _at(self, t) -> datetime:
        return datetime.combine(self.clock().date(), t, IST)

    def run(self) -> dict:
        now = self.clock()
        today = now.date()
        if now.weekday() >= 5:
            log.info("Weekend: no live session.")
            return {"status": "weekend"}
        if now >= self._at(CLOSE):
            log.info("Market already closed for today; nothing to do live.")
            return {"status": "after_close"}
        if self.acct.last_date and self.acct.last_date >= today.isoformat():
            log.info("Today (%s) is already processed in the paper account.", today)
            return {"status": "already_processed"}
        while self.clock() < self._at(OPEN):
            wait = (self._at(OPEN) - self.clock()).total_seconds()
            log.info("Waiting %.0f min for the 09:15 open...", wait / 60)
            self.sleep(min(max(wait, 1), 300))
        log.info("LIVE paper session started. Pending orders: %s. Prices: Yahoo 1-minute (delayed).",
                 list(self.acct.sim.pending) or "none")
        got_data = False
        while True:
            now = self.clock()
            closing = now >= self._at(CLOSE)
            bars = self.quotes(self.symbols)
            got_today = self._process(bars, now, today)
            got_data = got_data or got_today
            if not got_data and now >= self._at(OPEN) + timedelta(minutes=45):
                log.warning("No live bars for %s by 10:00. Holiday or feed outage: ending the live session.", today)
                self._status("no_data", now)
                return {"status": "no_data"}
            self._status("closed" if closing else "running", now)
            if closing:
                break
            self.sleep(self.poll)
        if not got_data:
            return {"status": "no_data"}
        ev = self.acct.live_close(pd.Timestamp(today), self.last_price)
        log.info("LIVE session closed for %s. Equity Rs %.2f. %s", today, self.acct.sim.total_equity, "; ".join(ev))
        return {"status": "closed", "equity": self.acct.sim.total_equity}

    def _process(self, bars: dict[str, pd.DataFrame], now: datetime, today) -> bool:
        """Handle every completed 1-minute bar not seen before. Returns True if today's data arrived."""
        got = False
        held = set(self.acct.sim.positions) | set(self.acct.sim.pending)
        for sym in self.symbols:
            df = bars.get(sym)
            if df is None or df.empty:
                continue
            df = df[(df.index.date == today) & (df.index <= now - timedelta(minutes=1))]  # completed bars only
            if df.empty:
                continue
            got = True
            seen = self.last_seen.get(sym)
            new = df if seen is None else df[df.index > seen]
            self.last_seen[sym] = df.index[-1]
            self.last_price[sym] = float(df["close"].iloc[-1])
            if sym not in held or new.empty:
                continue
            if sym in self.acct.sim.pending:
                first = new.index[0]
                if now - first.to_pydatetime() <= timedelta(minutes=LATE_FILL_MINUTES):
                    msg = self.acct.live_fill(sym, first, float(new["open"].iloc[0]), " (session open, delayed feed)")
                    # the fill happened at this bar's open, so its whole range counts for stops below
                else:
                    ts = new.index[-1]
                    msg = self.acct.live_fill(sym, ts, float(new["close"].iloc[-1]),
                                              f" (late start: filled at the {ts:%H:%M} price, not the open)")
                    new = new.iloc[0:0]  # filled at that bar's close: stops start with the next bar
                if msg:
                    log.info("LIVE %s", msg)
            for ts, b in new.iterrows():
                if sym not in self.acct.sim.positions:
                    break
                msg = self.acct.live_check(sym, ts, float(b["open"]), float(b["high"]), float(b["low"]))
                if msg:
                    log.info("LIVE %s", msg)
        self.acct.sim.mark(self.last_price)
        self.acct.save()
        return got

    def _status(self, state: str, now: datetime) -> None:
        self.acct.set_live_status({
            "state": state, "updated_at": now.isoformat(timespec="seconds"),
            "source": "Yahoo Finance 1-minute bars (unofficial, DELAYED)",
            "prices": {s: {"price": round(p, 2), "bar_time": self.last_seen[s].strftime("%H:%M")} for s, p in self.last_price.items()},
            "equity": round(self.acct.sim.total_equity, 2), "positions": len(self.acct.sim.positions),
            "pending": list(self.acct.sim.pending)})
