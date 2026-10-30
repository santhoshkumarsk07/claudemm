"""SQLite journal: every decision, order, fill, trade, equity snapshot and day."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .clock import now_ist
from .secure import owner_only, redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY, name TEXT, started_at TEXT, mode TEXT, feed TEXT, brain TEXT, model TEXT,
  starting_capital REAL, config_json TEXT, status TEXT DEFAULT 'active', ended_at TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, run_id INTEGER, ts TEXT, symbol TEXT, action TEXT, allocation_pct REAL,
  confidence REAL, stop_loss_pct REAL, target_pct REAL, reason TEXT, market_view TEXT, brain TEXT,
  outcome TEXT
);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY, run_id INTEGER, ts TEXT, decision_id INTEGER, symbol TEXT, side TEXT, qty INTEGER,
  quote_price REAL, fill_price REAL, slippage_cost REAL, charges REAL, charges_json TEXT,
  trigger TEXT, status TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY, run_id INTEGER, symbol TEXT, qty INTEGER, entry_ts TEXT, entry_price REAL,
  exit_ts TEXT, exit_price REAL, gross_pnl REAL, charges REAL, net_pnl REAL, exit_reason TEXT,
  hold_minutes REAL, entry_reason TEXT
);
CREATE TABLE IF NOT EXISTS positions (
  run_id INTEGER, symbol TEXT, qty INTEGER, avg_price REAL, stop REAL, target REAL, opened_at TEXT,
  entry_charges REAL, entry_reason TEXT, last_price REAL, PRIMARY KEY (run_id, symbol)
);
CREATE TABLE IF NOT EXISTS account (
  run_id INTEGER PRIMARY KEY, cash REAL, realized REAL, charges REAL, slippage REAL
);
CREATE TABLE IF NOT EXISTS equity (
  run_id INTEGER, ts TEXT, cash REAL, holdings REAL, equity REAL, realized REAL, unrealized REAL, charges REAL
);
CREATE TABLE IF NOT EXISTS days (
  run_id INTEGER, date TEXT, day_no INTEGER, start_equity REAL, end_equity REAL, pnl REAL, trades INTEGER,
  wins INTEGER, losses INTEGER, charges REAL, decisions INTEGER, insights TEXT, PRIMARY KEY (run_id, date)
);
CREATE TABLE IF NOT EXISTS notes (
  run_id INTEGER, ts TEXT, kind TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS day_state (
  run_id INTEGER, date TEXT, start_equity REAL, killed INTEGER DEFAULT 0, PRIMARY KEY (run_id, date)
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY, ts TEXT, run_id INTEGER, kind TEXT, data TEXT, prev_hash TEXT, hash TEXT
);
CREATE INDEX IF NOT EXISTS ix_equity_run ON equity(run_id, ts);
CREATE INDEX IF NOT EXISTS ix_dec_run ON decisions(run_id, ts);
"""

# Schema changes after this release, oldest first. Each entry is a tuple of SQL statements and
# runs once per database (tracked in PRAGMA user_version). Never edit or reorder a shipped entry.
MIGRATIONS: list[tuple[str, ...]] = []

GENESIS = "0" * 64


def _chain_hash(prev: str, ts: str, run_id, kind: str, data: str) -> str:
    return hashlib.sha256(f"{prev}|{ts}|{run_id}|{kind}|{data}".encode("utf-8")).hexdigest()


class Journal:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._tx_depth = 0
        owner_only(path)  # trading history and the audit trail are private
        self._migrate()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for n, statements in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.tx():
                for sql in statements:
                    self.x(sql)
                self.x(f"PRAGMA user_version = {n}")

    def x(self, sql: str, params=()) -> sqlite3.Cursor:
        cur = self.conn.execute(sql, params)
        if not self._tx_depth:
            self.conn.commit()
        return cur

    @contextmanager
    def tx(self):
        """Writes inside commit together or not at all. Nested blocks join the outer one."""
        self._tx_depth += 1
        try:
            if self._tx_depth == 1 and not self.conn.in_transaction:
                self.conn.execute("BEGIN IMMEDIATE")  # take the write lock now: reads inside see a stable state
            yield
        except BaseException:
            if self._tx_depth == 1:
                self.conn.rollback()
            raise
        else:
            if self._tx_depth == 1:
                self.conn.commit()
        finally:
            self._tx_depth -= 1

    def close(self) -> None:
        self.conn.close()

    def q(self, sql: str, params=()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()) -> dict | None:
        r = self.conn.execute(sql, params).fetchone()
        return dict(r) if r else None

    def new_run(self, name, mode, feed, brain, model, capital, cfg, started_at) -> int:
        cur = self.x(
            "INSERT INTO runs(name, started_at, mode, feed, brain, model, starting_capital, config_json) VALUES (?,?,?,?,?,?,?,?)",
            (name, started_at, mode, feed, brain, model, capital, json.dumps(cfg, default=str)),
        )
        self.x("INSERT INTO account VALUES (?,?,?,?,?)", (cur.lastrowid, capital, 0, 0, 0))
        return cur.lastrowid

    def active_run(self, mode: str) -> dict | None:
        return self.one("SELECT * FROM runs WHERE status='active' AND mode=? ORDER BY id DESC LIMIT 1", (mode,))

    def latest_run(self) -> dict | None:
        return self.one("SELECT * FROM runs ORDER BY id DESC LIMIT 1")

    # ---- audit trail -------------------------------------------------
    def audit(self, kind: str, data: dict, run_id: int | None = None, ts: str | None = None) -> str:
        """Append a tamper-evident event: each row's hash covers the previous row's hash,
        so editing or deleting any earlier row breaks every hash after it."""
        ts = ts or now_ist().isoformat(timespec="seconds")
        body = redact(json.dumps(data, sort_keys=True, default=str, ensure_ascii=False))
        with self.tx():
            last = self.one("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1")
            prev = last["hash"] if last else GENESIS
            h = _chain_hash(prev, ts, run_id, kind, body)
            self.x("INSERT INTO audit_log(ts, run_id, kind, data, prev_hash, hash) VALUES (?,?,?,?,?,?)",
                   (ts, run_id, kind, body, prev, h))
        return h

    def verify_audit(self) -> tuple[bool, int, str]:
        """Walk the chain. Returns (ok, rows checked, head hash or the first problem found)."""
        prev, n = GENESIS, 0
        for r in self.conn.execute("SELECT * FROM audit_log ORDER BY id"):
            if r["prev_hash"] != prev:
                return False, n, f"row {r['id']}: chain broken (a row before it was edited or deleted)"
            if _chain_hash(prev, r["ts"], r["run_id"], r["kind"], r["data"]) != r["hash"]:
                return False, n, f"row {r['id']}: contents were modified"
            prev, n = r["hash"], n + 1
        return True, n, prev
