"""Performance metrics. Sharpe and Sortino use daily returns, annualised with sqrt(252);
the risk-free rate defaults to 0 (set rf_annual, e.g. 0.065 for Indian T-bills)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def compute(equity: pd.Series, trades: list[dict], capital: float, charges: float = 0.0, slippage: float = 0.0,
            rf_annual: float = 0.0) -> dict:
    equity = equity.astype(float)
    if equity.empty:
        return {}
    rets = equity.pct_change().dropna()
    ex = rets - rf_annual / 252
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    final = float(equity.iloc[-1])
    sd, dsd = ex.std(ddof=1), math.sqrt((np.minimum(ex, 0) ** 2).mean()) if len(ex) else 0.0
    pnl = [t["net_pnl"] for t in trades]
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p <= 0]
    gw, gl = sum(wins), -sum(losses)
    held = sum(t.get("bars_held", 0) for t in trades)
    return {
        "start": str(equity.index[0])[:10], "end": str(equity.index[-1])[:10], "years": round(years, 2),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / capital - 1) * 100, 2),
        "cagr_pct": round(((final / capital) ** (1 / years) - 1) * 100, 2) if final > 0 else -100.0,
        "max_drawdown_pct": round(float(drawdown(equity).min()) * -100, 2),
        "sharpe": round(float(ex.mean() / sd * math.sqrt(252)), 2) if sd and sd > 0 else 0.0,
        "sortino": round(float(ex.mean() / dsd * math.sqrt(252)), 2) if dsd > 0 else 0.0,
        "trades": len(pnl),
        "win_rate_pct": round(len(wins) / len(pnl) * 100, 1) if pnl else 0.0,
        "profit_factor": round(gw / gl, 2) if gl > 0 else (math.inf if gw > 0 else 0.0),
        "avg_trade": round(sum(pnl) / len(pnl), 2) if pnl else 0.0,
        "avg_win": round(gw / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0.0,
        "avg_bars_held": round(held / len(pnl), 1) if pnl else 0.0,
        "charges_paid": round(charges, 2), "slippage_paid": round(slippage, 2),
        "costs_pct_of_capital": round((charges + slippage) / capital * 100, 2),
    }
