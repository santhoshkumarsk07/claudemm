"""Backtesting agent core: runs strategies through PortfolioSim on daily bars.

Bias controls
- Look-ahead: signals use data up to the close; fills happen at the next open (see sim.py).
  ML probabilities come only from walk-forward models trained on earlier data.
- Survivorship: today's NIFTY 50 list contains the survivors. Free point-in-time constituent
  lists are not available, so results on today's large caps are optimistic. The default
  watchlist leans on index ETFs (NIFTYBEES/BANKBEES) for that reason; report it with results.
- Costs: every fill pays slippage plus full Indian delivery charges.
"""
from __future__ import annotations

import pandas as pd

from . import metrics, strategies
from .risk import RiskConfig, RiskManager
from .sim import PortfolioSim


def run(data: dict[str, pd.DataFrame], strategy: str, cfg: dict, capital: float | None = None,
        p_up: dict[str, pd.Series] | None = None, start: str | None = None, end: str | None = None) -> dict:
    """Backtest one strategy on a portfolio of symbols sharing one cash balance.

    data: symbol -> indicator DataFrame (quant.indicators.add_all on adjusted bars).
    start/end: evaluation window. Indicators may warm up on bars before `start`.
    """
    capital = float(capital or cfg["paper"]["capital"])
    sigs = {}
    for s, d in data.items():
        if strategy in ("ml", "hybrid") and (not p_up or s not in p_up):
            continue
        sigs[s] = strategies.signals(strategy, d, (p_up or {}).get(s), cfg)
    if not sigs:
        raise ValueError(f"No symbols have inputs for strategy '{strategy}' (ML strategies need walk-forward predictions).")
    dates = sorted(set().union(*[data[s].index for s in sigs]))
    dates = [t for t in dates if (start is None or t >= pd.Timestamp(start)) and (end is None or t <= pd.Timestamp(end))]
    if not dates:
        raise ValueError("No bars inside the requested period.")
    risk = RiskManager(RiskConfig.from_cfg(cfg))
    sim = PortfolioSim(capital, risk, cfg["costs"]["slippage_bps"], cfg["costs"]["product"], cfg.get("charges"))
    for t in dates:
        bars = {s: data[s].loc[t] for s in sigs if t in data[s].index}
        sim.step(t, bars)
        if t == dates[-1]:
            break  # an order on the last day could never be filled inside the test
        for s in bars:
            entry, exit_ = sigs[s]
            if s in sim.positions and bool(exit_.get(t, False)):
                sim.order(s, "SELL", f"{strategy} exit", signal_date=t)
            elif s not in sim.positions and bool(entry.get(t, False)):
                sim.order(s, "BUY", strategies.explain(strategy, data[s], data[s].index.get_loc(t)),
                          atr=float(data[s].at[t, "atr14"]), signal_date=t)
    sim.close_all(dates[-1])
    eq = pd.Series(sim.equity)
    eq.index = pd.to_datetime(eq.index)
    m = metrics.compute(eq, sim.trades, capital, sim.charges_paid, sim.slippage_paid)
    m["exposure_pct"] = round(_exposure(sim.trades, dates) * 100, 1)
    m["rejected_by_risk"] = sum(1 for o in sim.orders if o["status"] == "rejected")
    return {"strategy": strategy, "symbols": list(sigs), "metrics": m, "equity": eq, "trades": sim.trades,
            "orders": sim.orders, "events": sim.events}


def buy_and_hold(data: dict[str, pd.DataFrame], cfg: dict, capital: float | None = None,
                 start: str | None = None, end: str | None = None) -> dict:
    """Benchmark: equal-weight buy at the first open, hold to the last close, with the same costs."""
    from . import costs
    capital = float(capital or cfg["paper"]["capital"])
    slip, prod = cfg["costs"]["slippage_bps"], cfg["costs"]["product"]
    per = capital / len(data)
    eq_parts, trades, chg_total, exit_adj = [], [], 0.0, 0.0
    for s, d in data.items():
        d = d[(d.index >= pd.Timestamp(start)) if start else slice(None)]
        d = d[(d.index <= pd.Timestamp(end)) if end else slice(None)]
        if d.empty:
            continue
        px = costs.fill_price("BUY", float(d["open"].iloc[0]), slip)
        qty = int(per * 0.995 // px)
        c_in = costs.charges("BUY", qty, px, prod)
        cash = per - qty * px - c_in
        eq_parts.append(cash + qty * d["close"])
        out = costs.fill_price("SELL", float(d["close"].iloc[-1]), slip)
        c_out = costs.charges("SELL", qty, out, prod)
        chg_total += c_in + c_out
        exit_adj += (out - float(d["close"].iloc[-1])) * qty - c_out  # sell at the last close, pay exit costs
        trades.append({"symbol": s, "net_pnl": (out - px) * qty - c_in - c_out, "bars_held": len(d)})
    eq = pd.concat(eq_parts, axis=1).ffill().sum(axis=1)
    eq.iloc[-1] += exit_adj
    m = metrics.compute(eq, trades, capital, chg_total)
    m["exposure_pct"] = 100.0
    return {"strategy": "buy_and_hold", "symbols": list(data), "metrics": m, "equity": eq, "trades": trades}


def _exposure(trades: list[dict], dates: list) -> float:
    if not dates:
        return 0.0
    days = set()
    for t in trades:
        days.update(pd.bdate_range(t["entry_date"], t["exit_date"]))
    return min(len(days & set(dates)) / len(dates), 1.0)
