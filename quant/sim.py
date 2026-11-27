"""Portfolio simulator shared by the backtester and the paper trader.

Timing rules (no look-ahead):
- A signal is decided on the CLOSE of day t, using only data up to t.
- It is filled at the OPEN of the next session (t+1), with slippage and charges.
- Stop-loss and take-profit are checked against each later day's high/low. If a day
  touches both, the stop is assumed to fill first (the conservative choice). A gap
  through the stop fills at the open, which can be worse than the stop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import costs
from .risk import RiskManager
from arena.safety import PAPER_ONLY


@dataclass
class Position:
    symbol: str
    qty: int
    entry_price: float
    entry_date: str
    stop: float
    target: float
    entry_charges: float
    reason: str = ""
    last_price: float = 0.0


class PortfolioSim:
    def __init__(self, capital: float, risk: RiskManager, slippage_bps: float = 5, product: str = "delivery",
                 rates: dict | None = None):
        assert PAPER_ONLY, "paper-only constant was tampered with"
        self.capital = float(capital)
        self.cash = float(capital)
        self.risk = risk
        self.slip, self.product, self.rates = slippage_bps, product, rates
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, dict] = {}  # symbol -> {"side", "reason", "atr", "signal_date"}
        self.trades: list[dict] = []
        self.orders: list[dict] = []
        self.equity: dict[str, float] = {}
        self.last_close: dict[str, float] = {}
        self.events: list[str] = []
        self.charges_paid = 0.0
        self.slippage_paid = 0.0

    # ---- state ---------------------------------------------------------
    @property
    def holdings(self) -> float:
        return sum(p.qty * self.last_close.get(s, p.entry_price) for s, p in self.positions.items())

    @property
    def total_equity(self) -> float:
        return self.cash + self.holdings

    def order(self, symbol: str, side: str, reason: str, atr: float | None = None, signal_date=None) -> None:
        """Queue an order for the next session's open (BUY opens, SELL closes; long-only)."""
        self.pending[symbol] = {"side": side, "reason": reason, "atr": atr,
                                "signal_date": str(signal_date)[:10] if signal_date is not None else None}

    # ---- one trading day -----------------------------------------------
    def step(self, date: pd.Timestamp, bars: dict[str, pd.Series]) -> list[str]:
        """Process one session from daily bars: fill queued orders at the open, run stops/targets,
        mark to the close. The live session does the same with 1-minute bars (see quant/live.py)."""
        ev = []
        for sym in list(self.pending):
            if sym in bars:  # no bar today (holiday/suspension): the order waits
                ev += [m for m in [self.fill_pending(sym, date, float(bars[sym]["open"]))] if m]
        for sym in list(self.positions):
            if sym in bars:
                b = bars[sym]
                ev += [m for m in [self.check_bar(sym, date, float(b["open"]), float(b["high"]), float(b["low"]))] if m]
        self.events += [f"{date.strftime('%Y-%m-%d')} {e}" for e in ev]
        return ev + self.close_session(date, {s: float(b["close"]) for s, b in bars.items()})

    def fill_pending(self, sym: str, ts: pd.Timestamp, price: float) -> str | None:
        """Fill a queued order at `price` (the session open, or the live price when it is placed)."""
        o = self.pending.pop(sym, None)
        if o is None:
            return None
        if o["side"] == "SELL" and sym in self.positions:
            return self._exit(sym, ts, price, "signal", o["reason"])
        if o["side"] == "BUY" and sym not in self.positions:
            return self._entry(sym, ts, price, o)
        return None

    def check_bar(self, sym: str, ts: pd.Timestamp, o: float, h: float, l: float) -> str | None:
        """Stop-loss / take-profit against one bar. Stop first if both are touched; gaps fill at the open."""
        p = self.positions.get(sym)
        if p is None:
            return None
        if l <= p.stop:
            return self._exit(sym, ts, min(o, p.stop), "stop_loss", f"low {l:.2f} <= stop {p.stop:.2f}")
        if h >= p.target:
            return self._exit(sym, ts, max(o, p.target), "take_profit", f"high {h:.2f} >= target {p.target:.2f}")
        return None

    def mark(self, prices: dict[str, float]) -> None:
        for sym, px in prices.items():
            self.last_close[sym] = float(px)
            if sym in self.positions:
                self.positions[sym].last_price = float(px)

    def close_session(self, date: pd.Timestamp, closes: dict[str, float]) -> list[str]:
        """End of session: mark to the closing prices, record equity, run the end-of-day risk checks."""
        self.mark(closes)
        d = date.strftime("%Y-%m-%d")
        eq = self.total_equity
        self.equity[d] = eq
        ev = self.risk.on_close(pd.Timestamp(d), eq)
        self.events += [f"{d} {e}" for e in ev]
        return ev

    def _entry(self, sym, date, open_px, o) -> str:
        d = date.strftime("%Y-%m-%d")
        px = costs.fill_price("BUY", open_px, self.slip)
        dec = self.risk.check_entry(sym, date, px, o["atr"] or 0.0, self.total_equity, self.cash, len(self.positions))
        if not dec.approved:
            self.orders.append({"date": d, "symbol": sym, "side": "BUY", "status": "rejected", "note": dec.text(), "signal_date": o["signal_date"]})
            return f"BUY {sym} rejected by risk manager: {'; '.join(dec.reasons)}"
        qty = dec.qty
        chg = costs.charges("BUY", qty, px, self.product, self.rates)
        while qty > 0 and qty * px + chg > self.cash:
            qty -= 1
            chg = costs.charges("BUY", qty, px, self.product, self.rates)
        if qty < 1:
            return f"BUY {sym} skipped: not enough cash after charges"
        self.cash -= qty * px + chg
        self.charges_paid += chg
        self.slippage_paid += (px - open_px) * qty
        # Stop/target keep the distances the risk manager approved, measured from the actual fill.
        self.positions[sym] = Position(sym, qty, px, d, dec.stop, dec.target, chg, o["reason"][:300], open_px)
        self.orders.append({"date": d, "symbol": sym, "side": "BUY", "qty": qty, "price": px, "charges": round(chg, 2),
                            "status": "filled", "note": dec.text(), "signal_date": o["signal_date"]})
        return f"BUY {qty} {sym} @ {px:.2f} (stop {dec.stop:.2f}, target {dec.target:.2f}, charges {chg:.2f})"

    def _exit(self, sym, date, raw_px, why, note) -> str:
        d, p = date.strftime("%Y-%m-%d"), self.positions.pop(sym)
        px = costs.fill_price("SELL", raw_px, self.slip)
        chg = costs.charges("SELL", p.qty, px, self.product, self.rates)
        self.cash += p.qty * px - chg
        self.charges_paid += chg
        self.slippage_paid += (raw_px - px) * p.qty
        gross = (px - p.entry_price) * p.qty
        net = gross - p.entry_charges - chg
        t = {"symbol": sym, "qty": p.qty, "entry_date": p.entry_date, "entry_price": p.entry_price, "exit_date": d,
             "exit_price": px, "stop": p.stop, "target": p.target, "gross_pnl": round(gross, 2),
             "charges": round(p.entry_charges + chg, 2), "net_pnl": round(net, 2),
             "return_pct": round(net / (p.entry_price * p.qty) * 100, 3), "exit_reason": why,
             "bars_held": len(pd.bdate_range(p.entry_date, d)) - 1, "entry_reason": p.reason}
        self.trades.append(t)
        self.orders.append({"date": d, "symbol": sym, "side": "SELL", "qty": p.qty, "price": px, "charges": round(chg, 2),
                            "status": "filled", "note": f"{why}: {note}"})
        self.risk.on_trade_closed(sym, date, net)
        return f"SELL {p.qty} {sym} @ {px:.2f} ({why}) net {net:+.2f}"

    def close_all(self, date: pd.Timestamp, why: str = "end_of_test") -> None:
        """Close open positions at the last close so results include them (backtests only)."""
        for sym in list(self.positions):
            self._exit(sym, date, self.last_close[sym], why, "closed at last close")
        self.equity[date.strftime("%Y-%m-%d")] = self.cash

    # ---- persistence ---------------------------------------------------
    def to_state(self) -> dict:
        return {"capital": self.capital, "cash": self.cash, "positions": {s: asdict(p) for s, p in self.positions.items()},
                "pending": self.pending, "last_close": self.last_close, "charges_paid": self.charges_paid,
                "slippage_paid": self.slippage_paid, "risk": self.risk.to_state()}

    def load_state(self, s: dict) -> None:
        self.capital, self.cash = s["capital"], s["cash"]
        self.positions = {k: Position(**v) for k, v in s["positions"].items()}
        self.pending, self.last_close = s["pending"], s["last_close"]
        self.charges_paid, self.slippage_paid = s.get("charges_paid", 0.0), s.get("slippage_paid", 0.0)
