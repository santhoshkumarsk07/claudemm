"""SQLite store. Every bar keeps the source it came from and when it was fetched."""
from __future__ import annotations

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..clock import IST

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL, adj_close REAL,
  volume REAL, source TEXT, fetched_at TEXT, PRIMARY KEY (symbol, interval, ts)
);
CREATE TABLE IF NOT EXISTS fetch_log (
  ts TEXT, symbol TEXT, interval TEXT, source TEXT, status TEXT, rows INTEGER, message TEXT
);
CREATE TABLE IF NOT EXISTS models (
  name TEXT PRIMARY KEY, trained_at TEXT, trained_through TEXT, meta_json TEXT, blob BLOB
);
CREATE TABLE IF NOT EXISTS predictions (
  symbol TEXT, model TEXT, ts TEXT, p_up REAL, kind TEXT, created_at TEXT, PRIMARY KEY (symbol, model, ts, kind)
);
CREATE TABLE IF NOT EXISTS backtests (
  id INTEGER PRIMARY KEY, run_at TEXT, symbol TEXT, strategy TEXT, start TEXT, end TEXT,
  metrics_json TEXT, equity_json TEXT, trades_json TEXT, config_json TEXT
);
"""


class SyntheticDataRefused(RuntimeError):
    pass


class Store:
    def __init__(self, path: str | Path, allow_synthetic: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self.allow_synthetic = allow_synthetic
        self.conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # dashboard can read while agents write
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    # ---- bars ----------------------------------------------------------
    def upsert_bars(self, df: pd.DataFrame, symbol: str, interval: str, source: str, fetched_at: datetime | None = None) -> int:
        if source.upper().startswith("SYNTHETIC") and not self.allow_synthetic:
            raise SyntheticDataRefused("Synthetic prices are for tests only and are never stored as market data.")
        fetched = (fetched_at or datetime.now(IST)).isoformat(timespec="seconds")
        fmt = (lambda t: t.strftime("%Y-%m-%d")) if interval == "1d" else (lambda t: t.isoformat())
        rows = [(symbol, interval, fmt(ts), r.open, r.high, r.low, r.close, r.adj_close, r.volume, source, fetched)
                for ts, r in df.iterrows()]
        self.conn.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def bars(self, symbol: str, interval: str = "1d", start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sql, p = "SELECT * FROM bars WHERE symbol=? AND interval=?", [symbol, interval]
        if start:
            sql += " AND ts>=?"; p.append(start)
        if end:
            sql += " AND ts<=?"; p.append(end)
        df = pd.read_sql_query(sql + " ORDER BY ts", self.conn, params=p)
        if df.empty:
            return df
        df.index = pd.to_datetime(df.pop("ts")) if interval == "1d" else pd.to_datetime(df.pop("ts"), utc=True).dt.tz_convert(IST)
        df.index.name = "ts"
        df.attrs.update(symbol=symbol, interval=interval, sources=sorted(df["source"].unique()),
                        fetched_at=df["fetched_at"].max())
        return df

    def symbols(self, interval: str = "1d") -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT symbol, COUNT(*) rows, MIN(ts) first, MAX(ts) last, GROUP_CONCAT(DISTINCT source) source, "
            "MAX(fetched_at) fetched_at FROM bars WHERE interval=? GROUP BY symbol ORDER BY symbol", (interval,))]

    def log_fetch(self, symbol, interval, source, status, rows=0, message="") -> None:
        self.conn.execute("INSERT INTO fetch_log VALUES (?,?,?,?,?,?,?)",
                          (datetime.now(IST).isoformat(timespec="seconds"), symbol, interval, source, status, rows, message[:500]))
        self.conn.commit()

    def fetch_log(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM fetch_log ORDER BY rowid DESC LIMIT ?", (limit,))]

    # ---- models, predictions, backtests --------------------------------
    def save_model(self, name: str, model, trained_through: str, meta: dict) -> None:
        self.conn.execute("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?)",
                          (name, datetime.now(IST).isoformat(timespec="seconds"), trained_through,
                           json.dumps(meta, default=str), pickle.dumps(model)))
        self.conn.commit()

    def load_model(self, name: str):
        r = self.conn.execute("SELECT * FROM models WHERE name=?", (name,)).fetchone()
        if not r:
            return None, None
        # Only load models this program saved into your own local database.
        return pickle.loads(r["blob"]), {**json.loads(r["meta_json"]), "trained_at": r["trained_at"], "trained_through": r["trained_through"]}

    def save_predictions(self, symbol: str, model: str, p: pd.Series, kind: str) -> None:
        now = datetime.now(IST).isoformat(timespec="seconds")
        self.conn.execute("DELETE FROM predictions WHERE symbol=? AND model=? AND kind=?", (symbol, model, kind))
        self.conn.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                              [(symbol, model, ts.strftime("%Y-%m-%d"), float(v), kind, now) for ts, v in p.dropna().items()])
        self.conn.commit()

    def predictions(self, symbol: str, model: str, kind: str = "walk_forward") -> pd.Series:
        df = pd.read_sql_query("SELECT ts, p_up FROM predictions WHERE symbol=? AND model=? AND kind=? ORDER BY ts",
                               self.conn, params=(symbol, model, kind))
        return pd.Series(df["p_up"].values, index=pd.to_datetime(df["ts"]), name="p_up")

    def save_backtest(self, symbol, strategy, start, end, metrics, equity: pd.Series, trades: list, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO backtests(run_at, symbol, strategy, start, end, metrics_json, equity_json, trades_json, config_json) VALUES (?,?,?,?,?,?,?,?,?)",
            (datetime.now(IST).isoformat(timespec="seconds"), symbol, strategy, str(start), str(end), json.dumps(metrics, default=float),
             json.dumps({k.strftime("%Y-%m-%d"): round(float(v), 2) for k, v in equity.items()}),
             json.dumps(trades, default=str), json.dumps(config, default=str)))
        self.conn.commit()
        return cur.lastrowid

    def latest_backtests(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute(
            "SELECT b.* FROM backtests b JOIN (SELECT symbol, strategy, MAX(id) id FROM backtests GROUP BY symbol, strategy) m "
            "ON b.id=m.id ORDER BY b.symbol, b.strategy")]
