"""Configuration: defaults below, overridden by quant.toml."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "data": {
        "indices": ["^NSEI", "^NSEBANK"],
        "watchlist": ["NIFTYBEES.NS", "BANKBEES.NS", "RELIANCE.NS", "HDFCBANK.NS"],
        "start": "2012-01-01",
        "db": "data/quant.db",
        "stale_after_hours": 30,
    },
    "costs": {"product": "delivery", "slippage_bps": 5},
    "risk": {
        "risk_per_trade_pct": 1.0, "max_position_pct": 25.0, "stop_atr_mult": 2.0,
        "take_profit_rr": 2.0, "min_reward_risk": 1.5, "max_daily_loss_pct": 2.0,
        "max_drawdown_pct": 15.0, "drawdown_pause_bars": 20, "cooldown_bars": 3, "max_open_positions": 5,
    },
    "ml": {
        "horizon": 5, "min_train_bars": 750, "test_bars": 63, "model": "gradient_boosting",
        "buy_threshold": 0.55, "hybrid_threshold": 0.50,
    },
    "paper": {"capital": 1_000_000, "db": "data/paper.db"},
    "llm": {"enabled": True, "host": "http://localhost:11434", "model": "llama3.1:8b", "timeout_seconds": 60},
    "safety": {"paper_only": True},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else ROOT / "quant.toml"
    cfg = _merge(DEFAULTS, tomllib.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else {})
    for section in ("data", "paper"):  # database paths are relative to the project folder
        if not Path(cfg[section]["db"]).is_absolute():
            cfg[section]["db"] = str(ROOT / cfg[section]["db"])
    return cfg
