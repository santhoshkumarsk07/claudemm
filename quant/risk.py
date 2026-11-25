"""Risk management. Deterministic code that no agent or LLM can bypass.

Every new entry asks RiskManager.check_entry(). It sizes the position from the
stop distance and refuses the trade when any limit is breached. Exits are never
blocked: reducing risk is always allowed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 25.0
    stop_atr_mult: float = 2.0
    take_profit_rr: float = 2.0
    min_reward_risk: float = 1.5
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 15.0
    drawdown_pause_bars: int = 20
    cooldown_bars: int = 3
    max_open_positions: int = 5

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RiskConfig":
        return cls(**{k: cfg["risk"][k] for k in cls.__dataclass_fields__ if k in cfg["risk"]})


@dataclass
class Decision:
    approved: bool
    qty: int = 0
    stop: float = 0.0
    target: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def text(self) -> str:
        return ("APPROVED " if self.approved else "REJECTED ") + "; ".join(self.reasons)


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.peak_equity: float | None = None
        self.prev_close_equity: float | None = None
        self.halted = False                 # max-drawdown circuit breaker
        self.halt_until: str | None = None
        self.blocked_until: str | None = None  # daily-loss block (ISO date, exclusive)
        self.cooldown: dict[str, str] = {}     # symbol -> first date it may be re-entered

    # ---- called by the simulator --------------------------------------
    def on_close(self, date: pd.Timestamp, equity: float) -> list[str]:
        """End-of-day update: drawdown halt and daily-loss block."""
        msgs = []
        if self.prev_close_equity and equity < self.prev_close_equity * (1 - self.cfg.max_daily_loss_pct / 100):
            nxt = (date + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
            self.blocked_until = (date + pd.offsets.BDay(2)).strftime("%Y-%m-%d")
            msgs.append(f"Daily loss above {self.cfg.max_daily_loss_pct}%: no new entries on {nxt}.")
        self.peak_equity = max(self.peak_equity or equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity * 100
        d = date.strftime("%Y-%m-%d")
        if self.halted and self.halt_until and d >= self.halt_until:
            # Circuit breaker: after the pause, restart from today's equity as the new high-water mark.
            # (Waiting for equity to recover would never end: a flat book cannot recover.)
            self.halted, self.halt_until, self.peak_equity = False, None, equity
            msgs.append(f"Drawdown pause over; trading resumes with Rs {equity:,.0f} as the new high-water mark.")
        elif not self.halted and dd >= self.cfg.max_drawdown_pct:
            self.halted = True
            self.halt_until = (date + pd.offsets.BDay(self.cfg.drawdown_pause_bars)).strftime("%Y-%m-%d")
            msgs.append(f"Drawdown {dd:.1f}% hit the {self.cfg.max_drawdown_pct}% limit: no new entries until {self.halt_until}.")
        self.prev_close_equity = equity
        return msgs

    def on_trade_closed(self, symbol: str, date: pd.Timestamp, net_pnl: float) -> None:
        if net_pnl < 0 and self.cfg.cooldown_bars > 0:
            self.cooldown[symbol] = (date + pd.offsets.BDay(self.cfg.cooldown_bars + 1)).strftime("%Y-%m-%d")

    def check_entry(self, symbol: str, date: pd.Timestamp, price: float, atr: float, equity: float,
                    cash: float, open_positions: int) -> Decision:
        d, c = date.strftime("%Y-%m-%d"), self.cfg
        if self.halted:
            return Decision(False, reasons=[f"max drawdown pause ({c.max_drawdown_pct}%) until {self.halt_until}"])
        if self.blocked_until and d < self.blocked_until:
            return Decision(False, reasons=["daily loss limit hit yesterday"])
        if symbol in self.cooldown and d < self.cooldown[symbol]:
            return Decision(False, reasons=[f"cooldown after a loss until {self.cooldown[symbol]}"])
        if open_positions >= c.max_open_positions:
            return Decision(False, reasons=[f"max {c.max_open_positions} open positions"])
        if not (atr > 0 and math.isfinite(atr)) or price <= 0:
            return Decision(False, reasons=["no valid ATR/price to place a stop"])
        risk_ps = c.stop_atr_mult * atr
        stop = price - risk_ps
        if stop <= 0:
            return Decision(False, reasons=["stop would be below zero"])
        target = price + c.take_profit_rr * risk_ps
        rr = (target - price) / risk_ps
        if rr < c.min_reward_risk:
            return Decision(False, reasons=[f"reward/risk {rr:.2f} below {c.min_reward_risk}"])
        by_risk = math.floor(equity * c.risk_per_trade_pct / 100 / risk_ps)
        by_cap = math.floor(equity * c.max_position_pct / 100 / price)
        by_cash = math.floor(cash * 0.995 / price)  # keep a little cash for charges
        qty = min(by_risk, by_cap, by_cash)
        if qty < 1:
            return Decision(False, reasons=["position size rounds to zero shares"])
        limit = {by_risk: "risk per trade", by_cap: "max position %", by_cash: "available cash"}[qty]
        return Decision(True, qty, round(stop, 2), round(target, 2),
                        [f"{qty} shares (limited by {limit}), stop {stop:.2f}, target {target:.2f}, R:R {rr:.1f}"])

    # ---- persistence (paper trading) -----------------------------------
    def to_state(self) -> dict:
        return {"cfg": asdict(self.cfg), "peak_equity": self.peak_equity, "prev_close_equity": self.prev_close_equity,
                "halted": self.halted, "halt_until": self.halt_until, "blocked_until": self.blocked_until, "cooldown": self.cooldown}

    @classmethod
    def from_state(cls, s: dict, cfg: RiskConfig) -> "RiskManager":
        rm = cls(cfg)
        for k in ("peak_equity", "prev_close_equity", "halted", "halt_until", "blocked_until", "cooldown"):
            setattr(rm, k, s.get(k, getattr(rm, k)))
        return rm
