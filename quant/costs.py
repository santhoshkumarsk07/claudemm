"""Transaction costs: Indian charges (from arena.charges, configurable rates) plus slippage."""
from __future__ import annotations

import math

from arena import charges as _charges

TICK = 0.05


def _tick(p: float, up: bool) -> float:
    n = p / TICK
    return round((math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)) * TICK, 2)


def fill_price(side: str, price: float, slippage_bps: float) -> float:
    """Adverse fill: buys pay up, sells receive less, rounded to the NSE tick."""
    if side == "BUY":
        return _tick(price * (1 + slippage_bps / 1e4), up=True)
    return _tick(price * (1 - slippage_bps / 1e4), up=False)


def charges(side: str, qty: int, price: float, product: str = "delivery", rates: dict | None = None) -> float:
    """Brokerage + STT + exchange + SEBI + stamp duty + GST (+ DP charge on delivery sells)."""
    return _charges.compute(side, qty, price, product, rates).total if qty > 0 else 0.0
