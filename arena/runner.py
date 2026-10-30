"""The trading loop. Same code drives live sessions, historical replay and the demo."""
from __future__ import annotations

import json
import logging
import time as _time
from datetime import datetime, timedelta

from . import insights, report
from .brain import RulesBrain, make_brain, normalize
from .clock import CLOSE, is_market_open, next_open, parse_hhmm
from .db import Journal
from .engine import PaperBroker
from .feeds.base import PriceFeed
from .indicators import snapshot

log = logging.getLogger("arena")


class Runner:
    def __init__(self, cfg: dict, feed: PriceFeed, journal: Journal, run_id: int, brain=None, report_path=None):
        self.cfg, self.feed, self.j, self.run_id = cfg, feed, journal, run_id
        self.st = cfg["strategy"]
        self.brain = brain or make_brain(cfg)
        self.fallback = RulesBrain(cfg)
        self.broker = PaperBroker(journal, run_id, cfg)
        self.universe = list(dict.fromkeys(cfg["market"]["watchlist"]))
        self.holidays = set(cfg["market"].get("holidays", []))
        self.report_path = report_path or cfg["paths"]["report"]
        self.square_off = parse_hhmm(self.st["square_off_time"])
        self.cutoff = parse_hhmm(self.st["no_new_entries_after"])
        self.day = None
        self.day_start_equity = None
        self.killed = False
        self.next_decision = None
        self.last_snapshot = None
        self.last_report = 0.0
        self.stop_requested = False  # set by the GUI's Stop button
        self._stale_logged = None

    def _sleep(self, seconds: float) -> None:
        """Live: sleep in 1 s slices so Stop takes effect quickly. Replay: advance the clock."""
        if not self.feed.live:
            self.feed.sleep(seconds)
            return
        end = _time.time() + seconds
        while not self.stop_requested and _time.time() < end:
            _time.sleep(min(1.0, end - _time.time()))

    # ---- day lifecycle ----------------------------------------------
    def days_done(self) -> int:
        return self.j.one("SELECT COUNT(*) n FROM days WHERE run_id=?", (self.run_id,))["n"]

    def open_day(self, now: datetime, quotes) -> None:
        self.day = now.date()
        d = self.day.isoformat()
        if self.st["product"] == "intraday":  # left open by a Stop before square-off: never carry them overnight
            for s in [s for s, p in self.broker.positions.items() if p.opened_at[:10] < d]:
                if s in quotes:
                    log.warning(self.broker.sell(now, quotes[s], "stale_square_off", "intraday position from an earlier session")[1])
        # Resuming the same day keeps the morning's baseline, so a restart cannot reset the daily loss limit.
        saved = self.j.one("SELECT start_equity, killed FROM day_state WHERE run_id=? AND date=?", (self.run_id, d))
        if saved:
            self.day_start_equity, self.killed = saved["start_equity"], bool(saved["killed"])
            log.info("=== Day %d  %s resumed: day started at Rs %.2f%s ===", self.days_done() + 1, self.day,
                     self.day_start_equity, ", daily loss limit already hit" if self.killed else "")
        else:
            self.day_start_equity, self.killed = self.broker.equity, False
            self._save_day_state()
            log.info("=== Day %d  %s  equity Rs %.2f ===", self.days_done() + 1, self.day, self.day_start_equity)
        self.next_decision = now

    def _save_day_state(self) -> None:
        self.j.x("INSERT OR REPLACE INTO day_state VALUES (?,?,?,?)",
                 (self.run_id, self.day.isoformat(), self.day_start_equity, int(self.killed)))

    def close_day(self, now: datetime) -> None:
        d = self.day.isoformat()
        if self.st["product"] == "intraday" and self.broker.positions:
            q = self._quotes(list(self.broker.positions), fresh_only=False)
            for m in self.broker.square_off_all(now, q, "square_off"):
                log.info(m)
        self.broker.snapshot(now)
        day_no = self.days_done() + 1
        stats = self.j.one(
            "SELECT COUNT(*) n, SUM(net_pnl>0) w, SUM(net_pnl<=0) l, COALESCE(SUM(charges),0) c FROM trades "
            "WHERE run_id=? AND substr(exit_ts,1,10)=?", (self.run_id, d))
        ndec = self.j.one("SELECT COUNT(*) n FROM decisions WHERE run_id=? AND substr(ts,1,10)=? AND action!='HOLD'", (self.run_id, d))["n"]
        end_eq = self.broker.equity
        notes = insights.day_insights(self.j, self.run_id, d, self.day_start_equity, end_eq)
        self.j.x("INSERT OR REPLACE INTO days VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                 (self.run_id, d, day_no, round(self.day_start_equity, 2), round(end_eq, 2),
                  round(end_eq - self.day_start_equity, 2), stats["n"], stats["w"] or 0, stats["l"] or 0,
                  round(stats["c"], 2), ndec, json.dumps(notes)))
        log.info("=== Day %d closed: P&L Rs %+.2f, equity Rs %.2f ===", day_no, end_eq - self.day_start_equity, end_eq)
        self.day = None
        self.write_report()

    # ---- helpers -----------------------------------------------------
    def _quotes(self, symbols: list[str], fresh_only: bool = True):
        for attempt in range(5):
            try:
                quotes = self.feed.quotes(symbols)
                break
            except Exception as e:  # keep the session alive through feed hiccups
                log.warning("Quote fetch failed (%s). Retrying...", e)
                if self.feed.live:
                    self._sleep(min(2 ** attempt, 30))
                if self.stop_requested:
                    return {}
        else:
            return {}
        if not fresh_only:
            return quotes
        # Old prices mean a market holiday or a frozen feed: never decide or fill on them.
        oldest = self.feed.now() - timedelta(minutes=self.st.get("max_quote_age_minutes", 15))
        fresh = {s: q for s, q in quotes.items() if q.ts >= oldest}
        if quotes and not fresh and self._stale_logged != self.feed.now().date():
            self._stale_logged = self.feed.now().date()
            log.warning("Prices are stale (latest %s). Market holiday or feed delay? Not trading on them.",
                        max(q.ts for q in quotes.values()).strftime("%d %b %H:%M"))
        return fresh

    def write_report(self) -> None:
        try:
            report.write(self.j, self.run_id, self.report_path, live=self.feed.live)
            self.last_report = _time.time()
        except Exception:
            log.exception("Report generation failed")

    def build_context(self, now: datetime, quotes) -> dict:
        market = {}
        for s in self.universe:
            if s in quotes:
                try:
                    bars = self.feed.candles(s, self.st["bar_minutes"], 3)
                except Exception as e:
                    log.warning("Candles for %s failed: %s", s, e)
                    bars = None
                market[s] = snapshot(quotes[s], bars)
        b = self.broker
        positions = [{
            "symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price, "ltp": p.last_price,
            "unrealized_pct": round((p.last_price / p.avg_price - 1) * 100, 2) if p.avg_price else 0,
            "stop": p.stop, "target": p.target,
            "held_minutes": round((now - datetime.fromisoformat(p.opened_at)).total_seconds() / 60),
        } for p in b.positions.values()]
        recent = self.j.q("SELECT symbol, net_pnl, exit_reason, hold_minutes FROM trades WHERE run_id=? ORDER BY id DESC LIMIT 6", (self.run_id,))
        lessons = [n for r in self.j.q("SELECT insights FROM days WHERE run_id=? ORDER BY date DESC LIMIT 3", (self.run_id,))
                   for n in json.loads(r["insights"] or "[]")[:3]]
        return {
            "time_ist": now.strftime("%Y-%m-%d %H:%M"),
            "challenge_day": f"{self.days_done() + 1} of {self.cfg['account']['challenge_days']}",
            "account": {
                "equity": round(b.equity, 2), "cash": round(b.cash, 2),
                "starting_capital": self.cfg["account"]["starting_capital"],
                "pnl_today": round(b.equity - self.day_start_equity, 2),
                "pnl_total": round(b.equity - self.cfg["account"]["starting_capital"], 2),
                "charges_paid": round(b.total_charges, 2),
            },
            "positions": positions,
            "entries_allowed": now.time() < self.cutoff and not self.killed,
            "market": market,
            "recent_trades": recent,
            "lessons_from_previous_days": lessons,
        }

    def decide_and_execute(self, now: datetime, quotes) -> None:
        ctx = self.build_context(now, quotes)
        allowed = set(ctx["market"]) | set(self.broker.positions)
        brain = self.brain
        try:
            out = normalize(brain.decide(ctx), allowed)
        except Exception as e:
            log.warning("%s failed (%s); using rules fallback for this cycle.", brain.name, e)
            self.j.audit("brain_fallback", {"brain": brain.name, "error": str(e)[:200]}, self.run_id, now.isoformat())
            brain = self.fallback
            out = normalize(brain.decide(ctx), allowed)
        view = out["market_view"]
        # Orders reach the market after the decision: re-quote before filling.
        if not self.feed.live:
            self.feed.sleep(self.cfg["execution"]["replay_latency_seconds"])
        exec_time = self.feed.now()
        fresh = self._quotes(self.universe + list(self.broker.positions)) or quotes
        self.broker.mark(fresh)
        # Sells first to free cash, then buys by confidence.
        decs = sorted(out["decisions"], key=lambda d: (d["action"] != "SELL", -d["confidence"]))
        for d in decs:
            sym, act = d["symbol"], d["action"]
            cur = self.j.x(
                "INSERT INTO decisions(run_id, ts, symbol, action, allocation_pct, confidence, stop_loss_pct, target_pct, reason, market_view, brain) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (self.run_id, now.isoformat(), sym, act, d["allocation_pct"], d["confidence"],
                 d["stop_loss_pct"], d["target_pct"], d["reason"], view, brain.name))
            did, msg = cur.lastrowid, "hold"
            q = fresh.get(sym)
            if act == "BUY":
                if not q:
                    msg = "rejected: no quote"
                elif not ctx["entries_allowed"]:
                    msg = "rejected: entries closed"
                elif d["confidence"] < self.st.get("min_confidence", 0):
                    msg = f"rejected: confidence below {self.st['min_confidence']}"
                else:
                    sl = d["stop_loss_pct"] if d["stop_loss_pct"] > 0 else self.st["default_stop_loss_pct"]
                    tg = d["target_pct"] if d["target_pct"] > 0 else self.st["default_target_pct"]
                    _, msg = self.broker.buy(exec_time, q, d["allocation_pct"], sl, tg, d["reason"], did)
            elif act == "SELL":
                if q and sym in self.broker.positions:
                    _, msg = self.broker.sell(exec_time, q, "ai", d["reason"], did)
                else:
                    msg = "ignored: not held"
            self.j.x("UPDATE decisions SET outcome=? WHERE id=?", (msg, did))
            if act != "HOLD":
                log.info("[%s] %s %s -> %s", brain.name, act, sym, msg)

    # ---- main loop ---------------------------------------------------
    def run(self) -> None:
        poll = self.st["poll_seconds"]
        target_days = self.cfg["account"]["challenge_days"]
        self.j.audit("session_start", {"feed": self.feed.name, "brain": self.brain.name, "days_done": self.days_done(),
                                       "product": self.st["product"], "profile": self.st.get("profile")},
                     self.run_id, self.feed.now().isoformat())
        while True:
            if self.stop_requested:
                self.broker.snapshot(self.feed.now())
                self.write_report()
                if self.st["product"] == "intraday" and self.broker.positions:
                    log.warning("%d intraday position(s) left open. They are squared off at 15:15 if you resume today, "
                                "or at the start of the next session.", len(self.broker.positions))
                log.info("Paused. Cash, positions and history are saved. Press Start to resume.")
                self.j.audit("session_paused", {"open_positions": len(self.broker.positions)}, self.run_id, self.feed.now().isoformat())
                return
            if self.feed.finished():
                break
            now = self.feed.now()
            if self.day and now.date() != self.day:
                self.close_day(now)
            if not is_market_open(now, self.holidays):
                if self.day:
                    self.close_day(now)
                if self.days_done() >= target_days:
                    break
                if self.feed.live:
                    nxt = next_open(now, self.holidays)
                    wait = (nxt - now).total_seconds()
                    log.info("Market closed. Next open %s (in %.0f min). Waiting...", nxt.strftime("%a %d %b %H:%M"), wait / 60)
                    self._sleep(min(max(wait, 5), 600))
                else:
                    self._sleep(poll)
                continue
            if self.day is None and self.days_done() >= target_days:
                break
            self.tick(now)
            self._sleep(poll)
        if self.day:
            self.close_day(self.feed.now())
        self.j.x("UPDATE runs SET status='complete', ended_at=? WHERE id=?", (self.feed.now().isoformat(), self.run_id))
        self.j.audit("run_complete", {"days": self.days_done(), "equity": round(self.broker.equity, 2)},
                     self.run_id, self.feed.now().isoformat())
        self.write_report()
        log.info("Run %d finished. Report: %s", self.run_id, self.report_path)

    def tick(self, now: datetime) -> None:
        symbols = list(dict.fromkeys(self.universe + list(self.broker.positions)))
        quotes = self._quotes(symbols)
        if not quotes:
            return
        if self.day is None:  # a day starts with its first fresh prices, so a holiday never counts as one
            self.open_day(now, quotes)
        b = self.broker
        b.mark(quotes)
        for m in b.check_exits(now, quotes):
            log.info(m)
        t = now.time()
        max_loss = self.st["max_daily_loss_pct"]
        if not self.killed and b.equity <= self.day_start_equity * (1 - max_loss / 100):
            self.killed = True
            self._save_day_state()
            self.j.audit("kill_switch", {"equity": round(b.equity, 2), "day_start_equity": round(self.day_start_equity, 2),
                                         "limit_pct": max_loss}, self.run_id, now.isoformat())
            log.warning("Daily loss limit (%.1f%%) hit. Closing positions, no new entries today.", max_loss)
            for m in b.square_off_all(now, quotes, "kill_switch"):
                log.info(m)
        if self.st["product"] == "intraday" and t >= self.square_off:
            for m in b.square_off_all(now, quotes, "square_off"):
                log.info(m)
        elif self.next_decision and now >= self.next_decision:
            self.decide_and_execute(now, quotes)
            self.next_decision = now + timedelta(minutes=self.st["decision_minutes"])
        if not self.last_snapshot or now - self.last_snapshot >= timedelta(minutes=5):
            b.snapshot(now)
            self.last_snapshot = now
        if self.feed.live and _time.time() - self.last_report > self.cfg["report"]["refresh_seconds"]:
            self.write_report()
