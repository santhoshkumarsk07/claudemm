"""Paper-trading account persisted in SQLite. Virtual money only; no broker code exists."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .clock import IST
from .risk import RiskConfig, RiskManager
from .sim import PortfolioSim

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS equity (date TEXT PRIMARY KEY, equity REAL, cash REAL, holdings REAL);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, json TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS agent_runs (id INTEGER PRIMARY KEY, ts TEXT, as_of TEXT, json TEXT);
"""


class PaperAccount:
    def __init__(self, cfg: dict, path: str | None = None):
        self.cfg = cfg
        self.path = path or cfg["paper"]["db"]
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        rc = RiskConfig.from_cfg(cfg)
        st = self._get("sim")
        risk = RiskManager.from_state(st["risk"], rc) if st else RiskManager(rc)
        self.sim = PortfolioSim(cfg["paper"]["capital"], risk, cfg["costs"]["slippage_bps"], cfg["costs"]["product"])
        if st:
            self.sim.load_state(st)
        self.last_date: str | None = self._get("last_date")
        self.mode: str = self._get("mode") or "live"

    def _get(self, k):
        r = self.conn.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else None

    def _put(self, k, v) -> None:
        self.conn.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (k, json.dumps(v, default=str)))

    def reset(self, capital: float | None = None, mode: str = "live") -> None:
        for t in ("state", "equity", "trades", "orders", "events", "agent_runs"):
            self.conn.execute(f"DELETE FROM {t}")
        self.conn.commit()
        cfg = dict(self.cfg)
        cfg["paper"] = {**self.cfg["paper"], "capital": capital or self.cfg["paper"]["capital"]}
        self.__init__(cfg, self.path)
        self.mode = mode
        self._put("mode", mode)
        self.save()

    def process_day(self, date: pd.Timestamp, bars: dict[str, pd.Series]) -> list[str]:
        """Fill queued orders at this session's open, run stops/targets, mark to the close."""
        n_tr, n_or = len(self.sim.trades), len(self.sim.orders)
        ev = self.sim.step(date, bars)
        d = date.strftime("%Y-%m-%d")
        self.conn.executemany("INSERT INTO trades(json) VALUES (?)", [(json.dumps(t),) for t in self.sim.trades[n_tr:]])
        self.conn.executemany("INSERT INTO orders(json) VALUES (?)", [(json.dumps(o),) for o in self.sim.orders[n_or:]])
        self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?,?,?)",
                          (d, round(self.sim.total_equity, 2), round(self.sim.cash, 2), round(self.sim.holdings, 2)))
        self.log(*[f"{d} {e}" for e in ev])
        self.last_date = d
        self.save()
        return ev

    def queue(self, symbol: str, side: str, reason: str, atr: float | None, signal_date) -> None:
        self.sim.order(symbol, side, reason, atr, signal_date)
        self.log(f"{str(signal_date)[:10]} queued {side} {symbol} for next open: {reason[:160]}")
        self.save()

    def log(self, *lines: str) -> None:
        now = datetime.now(IST).isoformat(timespec="seconds")
        self.conn.executemany("INSERT INTO events(ts, text) VALUES (?,?)", [(now, x) for x in lines])
        self.conn.commit()

    def save(self) -> None:
        self._put("sim", self.sim.to_state())
        self._put("last_date", self.last_date)
        self.conn.commit()

    def save_agent_run(self, as_of: str, report: dict) -> None:
        self.conn.execute("INSERT INTO agent_runs(ts, as_of, json) VALUES (?,?,?)",
                          (datetime.now(IST).isoformat(timespec="seconds"), as_of, json.dumps(report, default=str)))
        self.conn.commit()

    # ---- read side (dashboard, performance agent) ----------------------
    def equity_curve(self) -> pd.Series:
        df = pd.read_sql_query("SELECT date, equity FROM equity ORDER BY date", self.conn)
        return pd.Series(df["equity"].values, index=pd.to_datetime(df["date"]), dtype=float)

    def trades(self) -> list[dict]:
        return [json.loads(r["json"]) for r in self.conn.execute("SELECT json FROM trades ORDER BY id")]

    def orders(self, limit: int = 200) -> list[dict]:
        return [json.loads(r["json"]) for r in self.conn.execute("SELECT json FROM orders ORDER BY id DESC LIMIT ?", (limit,))]

    def events(self, limit: int = 200) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT ts, text FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    def last_agent_run(self) -> dict | None:
        r = self.conn.execute("SELECT * FROM agent_runs ORDER BY id DESC LIMIT 1").fetchone()
        return {"ts": r["ts"], "as_of": r["as_of"], **json.loads(r["json"])} if r else None
