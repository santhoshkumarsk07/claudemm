"""Run with: python -m unittest discover tests"""
from __future__ import annotations

import copy
import re
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arena import charges, safety  # noqa: E402
from arena.clock import IST  # noqa: E402
from arena.config import load_config  # noqa: E402
from arena.db import Journal  # noqa: E402
from arena.engine import PaperBroker  # noqa: E402
from arena.feeds.base import PriceFeed, Quote  # noqa: E402
from arena.feeds.sim import SimFeed  # noqa: E402
from arena.runner import Runner  # noqa: E402


def cfg_tmp(tmp: str) -> dict:
    cfg = load_config()
    cfg["paths"].update(db=f"{tmp}/t.db", report=f"{tmp}/r.html", data_dir=tmp, log=f"{tmp}/l.log")
    cfg["strategy"]["brain"] = "rules"
    return cfg


def q(sym="TEST", ltp=1000.0, bar_volume=0.0) -> Quote:
    return Quote(sym, ltp, ltp, ltp, ltp, ltp, 1e6, datetime(2026, 10, 5, 10, 0, tzinfo=IST), bar_volume)


class Charges(unittest.TestCase):
    def test_intraday_round_trip(self):
        b = charges.compute("BUY", 100, 1000, "intraday")
        s = charges.compute("SELL", 100, 1000, "intraday")
        self.assertAlmostEqual(b.brokerage, 20.0)          # 0.03% of 1L = 30, capped at 20
        self.assertAlmostEqual(b.stamp, 3.0)               # 0.003% buy side
        self.assertEqual(b.stt, 0)
        self.assertAlmostEqual(s.stt, 25.0)                # 0.025% sell side
        self.assertAlmostEqual(s.exchange, 2.97, places=2)
        self.assertAlmostEqual(s.gst, (20 + 2.97 + 0.1) * 0.18, places=2)

    def test_small_intraday_order_uses_percentage(self):
        self.assertAlmostEqual(charges.compute("BUY", 10, 1000, "intraday").brokerage, 3.0)

    def test_delivery(self):
        b = charges.compute("BUY", 100, 1000, "delivery")
        s = charges.compute("SELL", 100, 1000, "delivery")
        self.assertAlmostEqual(b.stt, 100.0)
        self.assertAlmostEqual(b.stamp, 15.0)
        self.assertAlmostEqual(s.dp, 15.93)


class Safety(unittest.TestCase):
    def test_order_urls_blocked(self):
        for m, url in [
            ("POST", "https://api.upstox.com/v2/order/place"),
            ("POST", "https://api.upstox.com/v3/order/place"),
            ("GET", "https://api.upstox.com/v2/portfolio/short-term-positions"),
            ("POST", "https://api.kite.trade/orders/regular"),
            ("POST", "https://apiconnect.angelone.in/rest/secure/angelbroking/order/v1/placeOrder"),
            ("GET", "https://example.com/anything"),
        ]:
            with self.assertRaises(safety.RealMoneyBlocked, msg=url):
                safety.check_url(m, url)

    def test_market_data_allowed(self):
        safety.check_url("GET", "https://api.upstox.com/v2/market-quote/quotes?instrument_key=NSE_EQ%7CINE002A01018")
        safety.check_url("GET", "https://api.upstox.com/v3/historical-candle/intraday/NSE_EQ%7CX/minutes/1")
        safety.check_url("GET", "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz")

    def test_broker_sdk_import_blocked(self):
        safety.install_guards()
        with self.assertRaises(safety.RealMoneyBlocked):
            __import__("kiteconnect")
        with self.assertRaises(safety.RealMoneyBlocked):
            __import__("upstox_client")

    def test_paper_only_cannot_be_disabled(self):
        with self.assertRaises(safety.RealMoneyBlocked):
            safety.install_guards({"safety": {"paper_only": False}})

    def test_no_order_code_in_package(self):
        pat = re.compile(r"place_?order|/order/|placeOrder|modify_order|cancel_order", re.I)
        for f in (ROOT / "arena").rglob("*.py"):
            if f.name == "safety.py":
                continue
            self.assertIsNone(pat.search(f.read_text()), f"order-like code in {f}")


class Engine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = cfg_tmp(self.tmp)
        self.j = Journal(self.cfg["paths"]["db"])
        self.rid = self.j.new_run("t", "test", "x", "rules", "", 100000, self.cfg, "2026-10-05T09:15")
        self.b = PaperBroker(self.j, self.rid, self.cfg)
        self.t = datetime(2026, 10, 5, 10, 0, tzinfo=IST)

    def test_fills_are_adverse_and_on_tick(self):
        buy, sell = self.b.fill_price("BUY", q(ltp=1000), 1), self.b.fill_price("SELL", q(ltp=1000), 1)
        self.assertGreater(buy, 1000)
        self.assertLess(sell, 1000)
        self.assertAlmostEqual(buy * 20, round(buy * 20))

    def test_buy_respects_position_cap_and_cash(self):
        ok, _ = self.b.buy(self.t, q(ltp=1000), 90, 1.5, 3, "test")
        self.assertTrue(ok)
        p = self.b.positions["TEST"]
        self.assertLessEqual(p.qty * p.avg_price, 100000 * self.cfg["strategy"]["max_position_pct"] / 100 + 1)
        self.assertGreaterEqual(self.b.cash, 0)

    def test_liquidity_cap(self):
        ok, _ = self.b.buy(self.t, q(ltp=100, bar_volume=200), 20, 1.5, 3, "thin")
        self.assertTrue(ok)
        self.assertLessEqual(self.b.positions["TEST"].qty, 200 * self.cfg["execution"]["max_participation"])

    def test_round_trip_accounting(self):
        self.b.buy(self.t, q(ltp=1000), 20, 1.5, 3, "in")
        self.b.sell(self.t, q(ltp=1010), "ai", "out")
        tr = self.j.q("SELECT * FROM trades")[0]
        self.assertAlmostEqual(self.b.cash - 100000, tr["net_pnl"], places=1)
        self.assertLess(tr["net_pnl"], tr["gross_pnl"])  # charges deducted
        self.assertLess(tr["gross_pnl"], 10 * tr["qty"])  # slippage: less than the naive Rs 10/share

    def test_stop_loss_triggers(self):
        self.b.buy(self.t, q(ltp=1000), 20, 1.0, 3, "in")
        msgs = self.b.check_exits(self.t, {"TEST": q(ltp=985)})
        self.assertTrue(msgs and "stop_loss" in msgs[0])
        self.assertEqual(self.b.positions, {})

    def test_bad_stop_and_target_are_clamped(self):
        self.b.buy(self.t, q(ltp=1000), 20, -1.0, 500, "bad levels")
        p = self.b.positions["TEST"]
        self.assertLess(p.stop, p.avg_price)                       # a negative stop is not placed above entry
        self.assertAlmostEqual(p.target, round(p.avg_price * 1.10, 2))  # capped at target_bounds[1] = 10%
        self.assertEqual(self.b.check_exits(self.t, {"TEST": q(ltp=1000)}), [])

    def test_failed_write_rolls_back(self):
        self.b.buy(self.t, q(ltp=1000), 20, 1.5, 3, "in")
        with self.assertRaises(RuntimeError):
            with self.j.tx():
                self.j.x("DELETE FROM positions WHERE run_id=?", (self.rid,))
                raise RuntimeError("crash mid-write")
        self.assertEqual(len(self.j.q("SELECT * FROM positions WHERE run_id=?", (self.rid,))), 1)

    def test_state_survives_restart(self):
        self.b.buy(self.t, q(ltp=1000), 20, 1.5, 3, "in")
        b2 = PaperBroker(self.j, self.rid, self.cfg)
        self.assertIn("TEST", b2.positions)
        self.assertAlmostEqual(b2.cash, self.b.cash)


class Replay(unittest.TestCase):
    def test_no_lookahead(self):
        f = SimFeed(["RELIANCE"], n_days=1, start=date(2026, 10, 5))
        for _ in range(20):
            now = f.now()
            qq = f.quotes(["RELIANCE"])["RELIANCE"]
            self.assertLess(qq.ts, now)
            self.assertTrue((f.candles("RELIANCE", 5, 3).index < now).all())
            f.sleep(60)


class FakeClaude:
    """Stands in for the API: buys the first two stocks it sees, then sells them."""
    name = "claude-api"

    def __init__(self):
        self.calls = 0

    def decide(self, ctx):
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("simulated API outage")
        held = {p["symbol"] for p in ctx["positions"]}
        decs = []
        for i, sym in enumerate(ctx["market"]):
            if sym in held and self.calls % 2 == 0:
                decs.append(dict(symbol=sym, action="SELL", allocation_pct=0, confidence=0.7, stop_loss_pct=0, target_pct=0, reason="take profit"))
            elif i < 2 and sym not in held:
                decs.append(dict(symbol=sym, action="BUY", allocation_pct=15, confidence=0.8, stop_loss_pct=1, target_pct=2, reason="test buy"))
            else:
                decs.append(dict(symbol=sym, action="HOLD", allocation_pct=0, confidence=0.5, stop_loss_pct=0, target_pct=0, reason="wait"))
        return {"market_view": "test view", "decisions": decs}


class EndToEnd(unittest.TestCase):
    def _run(self, brain=None, days=2):
        tmp = tempfile.mkdtemp()
        cfg = cfg_tmp(tmp)
        cfg["account"]["challenge_days"] = days
        feed = SimFeed(cfg["market"]["watchlist"][:4], n_days=days)
        cfg["market"]["watchlist"] = cfg["market"]["watchlist"][:4]
        j = Journal(cfg["paths"]["db"])
        rid = j.new_run("e2e", "demo", "sim", "rules", "", 100000, cfg, feed.now().isoformat())
        r = Runner(cfg, feed, j, rid, brain=brain)
        r.run()
        return j, rid, r, cfg

    def test_rules_two_days(self):
        j, rid, r, cfg = self._run()
        self.assertEqual(j.one("SELECT COUNT(*) n FROM days WHERE run_id=?", (rid,))["n"], 2)
        self.assertEqual(r.broker.positions, {})  # intraday: flat after square-off
        net = sum(t["net_pnl"] for t in j.q("SELECT net_pnl FROM trades WHERE run_id=?", (rid,)))
        self.assertAlmostEqual(r.broker.cash - 100000, net, places=1)
        self.assertEqual(j.one("SELECT status FROM runs WHERE id=?", (rid,))["status"], "complete")
        self.assertIn("2-day challenge", Path(cfg["paths"]["report"]).read_text())
        late = j.q("SELECT o.ts FROM orders o WHERE o.run_id=? AND o.side='BUY' AND o.status='filled' AND substr(o.ts,12,5) > ?",
                   (rid, cfg["strategy"]["no_new_entries_after"]))
        self.assertEqual(late, [])

    def test_claude_path_and_fallback(self):
        fake = FakeClaude()
        j, rid, r, _ = self._run(brain=fake, days=1)
        self.assertGreater(fake.calls, 3)
        brains = {x["brain"] for x in j.q("SELECT DISTINCT brain FROM decisions WHERE run_id=?", (rid,))}
        self.assertEqual(brains, {"claude-api", "rules"})  # outage cycle fell back to rules
        self.assertGreater(j.one("SELECT COUNT(*) n FROM orders WHERE run_id=? AND status='filled'", (rid,))["n"], 0)


class Adapters(unittest.TestCase):
    """Parsing checks for the live adapters with canned responses (no network)."""

    def test_yahoo_parsing(self):
        import pandas as pd
        import yfinance
        from arena.feeds.yahoo import YahooFeed

        idx = pd.date_range("2026-10-05 03:45", periods=4, freq="1min", tz="UTC").append(
            pd.date_range("2026-10-06 03:45", periods=3, freq="1min", tz="UTC"))
        cols = pd.MultiIndex.from_product([["RELIANCE.NS", "TCS.NS"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
        raw = pd.DataFrame(1000.0, index=idx, columns=cols)
        raw[("TCS.NS", "Close")] = [3000, 3001, 3002, 3003, 3010, 3011, 3012]
        orig = yfinance.download
        yfinance.download = lambda *a, **k: raw
        try:
            qs = YahooFeed().quotes(["RELIANCE", "TCS"])
        finally:
            yfinance.download = orig
        self.assertEqual(qs["TCS"].ltp, 3012)
        self.assertEqual(qs["TCS"].prev_close, 3003)
        self.assertEqual(qs["TCS"].ts.hour, 9)  # converted to IST

    def test_upstox_parsing(self):
        import json as _json
        from arena.feeds.upstox import UpstoxFeed

        class Resp:
            def __init__(self, data): self._d = data
            def raise_for_status(self): pass
            def json(self): return self._d

        class FakeHTTP:
            def get(self, url, **kw):
                safety.check_url("GET", url)
                if "market-quote/quotes" in url:
                    return Resp({"status": "success", "data": {"NSE_EQ:RELIANCE": {
                        "ohlc": {"open": 1440, "high": 1460, "low": 1435, "close": 1450},
                        "last_price": 1452.5, "volume": 3000000, "net_change": 12.5,
                        "instrument_token": "NSE_EQ|INE002A01018"}}})
                return Resp({"data": {"candles": [["2026-10-06T09:20:00+05:30", 1, 2, 0.5, 1.5, 100, 0],
                                                  ["2026-10-06T09:15:00+05:30", 1, 2, 0.5, 1.2, 100, 0]]}})

        tmp = tempfile.mkdtemp()
        Path(tmp, "upstox_nse_instruments.json").write_text(_json.dumps([
            {"segment": "NSE_EQ", "instrument_type": "EQ", "trading_symbol": "RELIANCE", "instrument_key": "NSE_EQ|INE002A01018"}]))
        f = UpstoxFeed(access_token="x", cache_dir=tmp)
        f.http = FakeHTTP()
        q_ = f.quotes(["RELIANCE"])["RELIANCE"]
        self.assertEqual(q_.ltp, 1452.5)
        self.assertEqual(q_.prev_close, 1440.0)
        bars = f.minute_bars("RELIANCE", 1)
        self.assertTrue(bars.index.is_monotonic_increasing)
        self.assertEqual(float(bars["close"].iloc[-1]), 1.5)


class Profiles(unittest.TestCase):
    def test_aggressive_overrides_and_confidence_gate(self):
        from arena.config import apply_profile
        cfg = load_config()
        cfg["strategy"]["profile"] = "aggressive"
        apply_profile(cfg)
        self.assertEqual(cfg["strategy"]["max_positions"], 3)
        self.assertEqual(cfg["strategy"]["max_position_pct"], 35)
        self.assertGreaterEqual(cfg["strategy"]["min_confidence"], 0.75)

    def test_low_confidence_buy_rejected(self):
        class Timid(FakeClaude):
            def decide(self, ctx):
                out = super().decide(ctx)
                for d in out["decisions"]:
                    d["confidence"] = 0.5
                return out
        tmp = tempfile.mkdtemp()
        cfg = cfg_tmp(tmp)
        cfg["strategy"]["profile"] = "aggressive"
        from arena.config import apply_profile
        apply_profile(cfg)
        cfg["account"]["challenge_days"] = 1
        cfg["market"]["watchlist"] = cfg["market"]["watchlist"][:3]
        feed = SimFeed(cfg["market"]["watchlist"], n_days=1)
        j = Journal(cfg["paths"]["db"])
        rid = j.new_run("p", "demo", "sim", "rules", "", 100000, cfg, feed.now().isoformat())
        Runner(cfg, feed, j, rid, brain=Timid()).run()
        self.assertEqual(j.q("SELECT COUNT(*) n FROM orders WHERE side='BUY' AND status='filled'")[0]["n"], 0)
        self.assertGreater(j.q("SELECT COUNT(*) n FROM decisions WHERE outcome LIKE 'rejected: confidence%'")[0]["n"], 0)


class BrainOutput(unittest.TestCase):
    def test_normalize_cleans_and_drops(self):
        from arena.brain import normalize
        out = normalize({"market_view": 5, "decisions": [
            {"symbol": "tcs", "action": "buy", "confidence": None, "allocation_pct": float("nan"), "stop_loss_pct": "1.2"},
            {"symbol": "NOTINLIST", "action": "BUY"},
            {"symbol": "INFY", "action": "<img src=x onerror=alert(1)>"},
            "junk",
        ]}, {"TCS", "INFY"})
        self.assertEqual(len(out["decisions"]), 1)
        d = out["decisions"][0]
        self.assertEqual((d["symbol"], d["action"], d["confidence"], d["allocation_pct"], d["stop_loss_pct"]),
                         ("TCS", "BUY", 0.0, 0.0, 1.2))
        self.assertEqual(out["market_view"], "5")

    def test_unusable_reply_raises(self):
        from arena.brain import normalize
        for bad in (None, [], {"decisions": "x"}, {"decisions": [{"symbol": "TCS"}]}):
            with self.assertRaises(ValueError):
                normalize(bad, {"TCS"})

    def test_malformed_replies_do_not_crash_the_run(self):
        class Sloppy:
            name = "claude-cli"

            def __init__(self):
                self.calls = 0

            def decide(self, ctx):
                self.calls += 1
                sym = next(iter(ctx["market"]))
                return [
                    {"market_view": "x", "decisions": [{"symbol": sym, "confidence": 0.9}]},  # no action
                    {"market_view": "x", "decisions": [{"symbol": sym, "action": "BUY", "confidence": None}]},
                    {"decisions": "not a list"},
                ][self.calls % 3]

        tmp = tempfile.mkdtemp()
        cfg = cfg_tmp(tmp)
        cfg["account"]["challenge_days"] = 1
        cfg["market"]["watchlist"] = cfg["market"]["watchlist"][:2]
        feed = SimFeed(cfg["market"]["watchlist"], n_days=1)
        j = Journal(cfg["paths"]["db"])
        rid = j.new_run("s", "demo", "sim", "rules", "", 100000, cfg, feed.now().isoformat())
        Runner(cfg, feed, j, rid, brain=Sloppy()).run()
        self.assertEqual(j.one("SELECT status FROM runs WHERE id=?", (rid,))["status"], "complete")
        brains = {r["brain"] for r in j.q("SELECT DISTINCT brain FROM decisions WHERE run_id=?", (rid,))}
        self.assertEqual(brains, {"claude-cli", "rules"})  # usable replies kept, unusable ones fell back


class Report(unittest.TestCase):
    def test_calls_are_escaped(self):
        from arena import report
        cfg = cfg_tmp(tempfile.mkdtemp())
        j = Journal(cfg["paths"]["db"])
        rid = j.new_run("r", "demo", "sim", "rules", "", 100000, cfg, "2026-10-05T09:15:00+05:30")
        j.x("INSERT INTO decisions(run_id, ts, symbol, action, confidence, reason) VALUES (?,?,?,?,?,?)",
            (rid, "2026-10-05T10:00:00+05:30", "TCS", "<img src=x onerror=alert(1)>", 0.5, "<script>x</script>"))
        page = report.write(j, rid, cfg["paths"]["report"]).read_text(encoding="utf-8")
        self.assertNotIn("<img src=x", page)
        self.assertNotIn("<script>x</script>", page)


class DayLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = cfg_tmp(self.tmp)
        self.cfg["market"]["watchlist"] = ["RELIANCE"]
        self.j = Journal(self.cfg["paths"]["db"])
        self.rid = self.j.new_run("d", "demo", "sim", "rules", "", 100000, self.cfg, "2026-10-05T09:15:00+05:30")

    def runner(self, feed=None):
        return Runner(self.cfg, feed or SimFeed(["RELIANCE"], n_days=2), self.j, self.rid, report_path=f"{self.tmp}/r.html")

    def test_restart_keeps_day_baseline_and_kill_switch(self):
        t = datetime(2026, 10, 5, 11, 0, tzinfo=IST)
        r1 = self.runner()
        r1.open_day(t, {})
        r1.killed = True
        r1._save_day_state()
        self.j.x("UPDATE account SET cash=95000 WHERE run_id=?", (self.rid,))
        r2 = self.runner()
        r2.open_day(t.replace(hour=13), {})
        self.assertEqual(r2.day_start_equity, 100000)
        self.assertTrue(r2.killed)

    def test_intraday_position_is_not_carried_overnight(self):
        r1 = self.runner()
        r1.broker.buy(datetime(2026, 10, 5, 14, 0, tzinfo=IST), q("RELIANCE", 1450), 20, 1.5, 3, "in")
        r2 = self.runner()
        r2.open_day(datetime(2026, 10, 6, 9, 20, tzinfo=IST), {"RELIANCE": q("RELIANCE", 1440)})
        self.assertEqual(r2.broker.positions, {})
        self.assertEqual(self.j.one("SELECT exit_reason FROM trades WHERE run_id=?", (self.rid,))["exit_reason"], "stale_square_off")

    def test_stale_prices_do_not_open_a_day(self):
        class Frozen(PriceFeed):  # e.g. an NSE holiday: the feed keeps serving the last session's prices
            live = False

            def now(self):
                return datetime(2026, 10, 6, 11, 0, tzinfo=IST)

            def quotes(self, symbols):
                return {s: q(s, 1450) for s in symbols}  # stamped 2026-10-05 10:00

            def candles(self, *a, **k):
                raise AssertionError("no analysis on stale prices")

        feed = Frozen()
        r = self.runner(feed)
        r.tick(feed.now())
        self.assertIsNone(r.day)
        self.assertEqual(self.j.one("SELECT COUNT(*) n FROM decisions WHERE run_id=?", (self.rid,))["n"], 0)


class Panel(unittest.TestCase):
    def setUp(self):
        import threading
        from http.server import ThreadingHTTPServer
        from arena.gui import Controller, make_handler
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Controller(), "s3cret"))
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def request(self, method, path, headers, body=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        c.request(method, path, body=body, headers=headers)
        r = c.getresponse()
        out = r.status, r.read()
        c.close()
        return out

    def test_actions_need_token_and_local_origin(self):
        ok = {"Content-Type": "application/json", "X-Arena-Token": "s3cret"}
        post = lambda h: self.request("POST", "/api/stop", h, "{}")
        self.assertEqual(post({"Content-Type": "text/plain"})[0], 403)        # cross-site "simple" request
        self.assertEqual(post({**ok, "X-Arena-Token": "wrong"})[0], 403)
        self.assertEqual(post({**ok, "Origin": "https://evil.example"})[0], 403)
        self.assertEqual(post({**ok, "Host": "evil.example"})[0], 403)       # DNS rebinding
        status, body = post({**ok, "Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)
        self.assertIn(b"Nothing is running", body)

    def test_reads_need_local_host_and_panel_embeds_token(self):
        self.assertEqual(self.request("GET", "/api/state", {"Host": "evil.example"})[0], 403)
        status, page = self.request("GET", "/", {})
        self.assertEqual(status, 200)
        self.assertIn(b'const TOKEN="s3cret"', page)



class Secrets(unittest.TestCase):
    def test_logs_are_redacted(self):
        import io
        import logging
        import os
        from arena.secure import RedactingFormatter
        os.environ["UPSTOX_ACCESS_TOKEN"] = "eyJ0eXAiOiJKV1QiLCJhbGciOi.secret"
        try:
            buf = io.StringIO()
            h = logging.StreamHandler(buf)
            h.setFormatter(RedactingFormatter("%(message)s"))
            lg = logging.getLogger("redact-test")
            lg.addHandler(h)
            lg.propagate = False
            lg.warning("token %s", os.environ["UPSTOX_ACCESS_TOKEN"])
            lg.warning("headers {'Authorization': 'Bearer abcdefghijklmnop'} key sk-ant-api03-ZZZZZZZZZZZZ")
            lg.warning("POST /token code=4f9a2b7c client_secret=hunter22hunter grant_type=authorization_code")
            try:
                raise RuntimeError("failed with access_token=SHOULDNOTLEAK123")
            except RuntimeError:
                lg.exception("boom")
            out = buf.getvalue()
        finally:
            del os.environ["UPSTOX_ACCESS_TOKEN"]
        for leaked in ("eyJ0eXAiOiJKV1Qi", "abcdefghijklmnop", "sk-ant-api03", "4f9a2b7c", "hunter22", "SHOULDNOTLEAK"):
            self.assertNotIn(leaked, out)
        self.assertIn("max_tokens=16000", __import__("arena.secure", fromlist=["redact"]).redact("max_tokens=16000"))

    @unittest.skipUnless(sys.platform != "win32", "POSIX permissions")
    def test_env_and_journal_are_owner_only(self):
        import os
        import stat
        from arena.config import set_env_value
        tmp = Path(tempfile.mkdtemp())
        env = tmp / ".env"
        env.write_text("A=1\n")
        os.chmod(env, 0o644)
        set_env_value("UPSTOX_ACCESS_TOKEN", "x" * 20, env)
        os.environ.pop("UPSTOX_ACCESS_TOKEN", None)
        self.assertEqual(stat.S_IMODE(env.stat().st_mode), 0o600)
        Journal(tmp / "j.db")
        self.assertEqual(stat.S_IMODE((tmp / "j.db").stat().st_mode), 0o600)


class Audit(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_tmp(tempfile.mkdtemp())
        self.j = Journal(self.cfg["paths"]["db"])
        self.rid = self.j.new_run("a", "test", "x", "rules", "", 100000, self.cfg, "2026-10-05T09:15")

    def test_orders_are_audited_and_chain_verifies(self):
        b = PaperBroker(self.j, self.rid, self.cfg)
        t = datetime(2026, 10, 5, 10, 0, tzinfo=IST)
        b.buy(t, q(ltp=1000), 20, 1.5, 3, "in")
        b.sell(t, q(ltp=1010), "ai", "out")
        b.buy(t, q(ltp=1000), 0, 1.5, 3, "zero allocation")  # rejected orders are audited too
        kinds = [r["kind"] for r in self.j.q("SELECT kind FROM audit_log ORDER BY id")]
        self.assertEqual(kinds, ["order", "order", "order"])
        ok, n, head = self.j.verify_audit()
        self.assertTrue(ok)
        self.assertEqual((n, len(head)), (3, 64))

    def test_tampering_is_detected(self):
        for i in range(3):
            self.j.audit("note", {"i": i}, self.rid)
        self.j.x("UPDATE audit_log SET data=? WHERE id=2", ('{"i": 99}',))
        ok, n, why = self.j.verify_audit()
        self.assertFalse(ok)
        self.assertIn("row 2", why)
        self.j.x("DELETE FROM audit_log WHERE id=2")  # removing the row breaks the link instead
        self.assertIn("chain broken", self.j.verify_audit()[2])

    def test_secrets_never_enter_the_audit_trail(self):
        self.j.audit("note", {"msg": "Bearer abcdefghijklmnop"}, self.rid)
        self.assertNotIn("abcdefghijklmnop", self.j.one("SELECT data FROM audit_log")["data"])
        self.assertTrue(self.j.verify_audit()[0])

    def test_run_lifecycle_is_audited(self):
        feed = SimFeed(["RELIANCE"], n_days=1)
        self.cfg["market"]["watchlist"] = ["RELIANCE"]
        self.cfg["account"]["challenge_days"] = 1
        Runner(self.cfg, feed, self.j, self.rid).run()
        kinds = {r["kind"] for r in self.j.q("SELECT kind FROM audit_log")}
        self.assertTrue({"session_start", "run_complete"} <= kinds)
        self.assertTrue(self.j.verify_audit()[0])


class Migrations(unittest.TestCase):
    def test_each_migration_runs_once(self):
        from arena import db
        path = Path(tempfile.mkdtemp()) / "m.db"
        Journal(path).close()  # an existing database from before the migration
        saved = list(db.MIGRATIONS)
        db.MIGRATIONS.append(("ALTER TABLE trades ADD COLUMN setup TEXT",))
        try:
            j = Journal(path)
            j2 = Journal(path)  # reopening must not apply it again (would fail: duplicate column)
            self.assertEqual(j2.conn.execute("PRAGMA user_version").fetchone()[0], len(db.MIGRATIONS))
            self.assertIn("setup", [c["name"] for c in j.q("PRAGMA table_info(trades)")])
        finally:
            db.MIGRATIONS[:] = saved


if __name__ == "__main__":
    unittest.main()
