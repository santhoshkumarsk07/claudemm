"""Tests for the quant package. Run: python -m unittest discover tests

All tests use SYNTHETIC prices (labelled as such) so they run offline and deterministically.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from quant import agents, backtest, config, indicators, pipeline, strategies
from quant.data import quality, synthetic
from quant.data.store import Store, SyntheticDataRefused
from quant.llm import Ollama
from quant.ml import features, walkforward
from quant.paper import PaperAccount
from quant.risk import RiskConfig, RiskManager
from quant.sim import PortfolioSim


def tmp_cfg(d: str) -> dict:
    cfg = config.load()
    cfg["data"].update(db=f"{d}/q.db", indices=["^NSEI"], watchlist=["AAA.NS", "BBB.NS"])
    cfg["paper"].update(db=f"{d}/p.db", capital=1_000_000)
    cfg["ml"].update(min_train_bars=400, test_bars=100, model="logistic")
    cfg["llm"]["enabled"] = False
    return cfg


def seeded_store(cfg, n=900) -> Store:
    s = Store(cfg["data"]["db"], allow_synthetic=True)
    for i, sym in enumerate(["^NSEI", "AAA.NS", "BBB.NS"]):
        s.upsert_bars(synthetic.daily_bars(n, seed=i + 3), sym, "1d", synthetic.SOURCE)
    return s


class Indicators(unittest.TestCase):
    def setUp(self):
        self.df = synthetic.daily_bars(600, seed=2)

    def test_known_values(self):
        c = self.df["close"]
        m = indicators.macd(c)
        pd.testing.assert_series_equal(m["macd"], indicators.ema(c, 12) - indicators.ema(c, 26), check_names=False)
        b = indicators.bollinger(c)
        pd.testing.assert_series_equal(b["bb_mid"], c.rolling(20).mean(), check_names=False)
        r = indicators.rsi(c).dropna()
        self.assertTrue(((r >= 0) & (r <= 100)).all())
        self.assertAlmostEqual(indicators.rsi(pd.Series(np.arange(1.0, 60))).iloc[-1], 100.0)  # only gains

    def test_atr_adx_vwap(self):
        n = 120
        up = pd.DataFrame({"open": np.arange(n) + 100.0, "close": np.arange(n) + 100.5, "volume": 1000.0}, index=pd.bdate_range("2020-01-01", periods=n))
        up["high"], up["low"] = up["close"] + 0.5, up["open"] - 0.5
        self.assertAlmostEqual(indicators.atr(up).iloc[-1], 1.5, places=2)       # constant true range
        self.assertGreater(indicators.adx(up)["adx"].iloc[-1], 60)               # strong steady trend
        flat = up.assign(high=10.0, low=10.0, close=10.0)
        self.assertAlmostEqual(indicators.vwap(flat, 20).iloc[-1], 10.0)
        self.assertTrue(indicators.vwap(flat.assign(volume=0.0), 20).isna().all())  # no volume -> no VWAP

    def test_indicators_and_features_are_causal(self):
        """Values at bar t must not change when later bars are removed (no look-ahead)."""
        full = indicators.add_all(self.df)
        ffull = features.build(full)
        for t in (250, 400, 599):
            part = indicators.add_all(self.df.iloc[: t + 1])
            pd.testing.assert_series_equal(full.iloc[t], part.iloc[-1], check_names=False, rtol=1e-9)
            pd.testing.assert_series_equal(ffull.iloc[t], features.build(part).iloc[-1], check_names=False, rtol=1e-9)
            for name in strategies.TRADITIONAL:
                e1, x1 = strategies.signals(name, full)
                e2, x2 = strategies.signals(name, part)
                self.assertEqual((e1.iloc[t], x1.iloc[t]), (e2.iloc[-1], x2.iloc[-1]), name)

    def test_label_alignment(self):
        d = self.df
        fwd = features.forward_return(d, 5)
        self.assertAlmostEqual(fwd.iloc[10], d["open"].iloc[16] / d["open"].iloc[11] - 1)
        self.assertTrue(fwd.iloc[-6:].isna().all())  # unknown future is NaN, never filled


class Data(unittest.TestCase):
    def test_store_refuses_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SyntheticDataRefused):
                Store(f"{d}/x.db").upsert_bars(synthetic.daily_bars(10), "X", "1d", synthetic.SOURCE)

    def test_store_keeps_source_and_timestamp(self):
        with tempfile.TemporaryDirectory() as d:
            s = Store(f"{d}/x.db", allow_synthetic=True)
            s.upsert_bars(synthetic.daily_bars(30), "X", "1d", "yahoo")
            b = s.bars("X")
            self.assertEqual(len(b), 30)
            self.assertEqual(b.attrs["sources"], ["yahoo"])
            self.assertTrue(b.attrs["fetched_at"])

    def test_bad_prints_removed_real_moves_kept(self):
        df = synthetic.daily_bars(60, seed=1)
        df.iloc[20, :4] = df.iloc[20, :4] / 10      # 1/10-scale feed error that reverts
        df.iloc[40:, :4] = df.iloc[40:, :4] * 0.7   # a genuine 30% crash that stays
        clean, issues = quality.clean(df)
        self.assertNotIn(df.index[20], clean.index)
        self.assertIn(df.index[40], clean.index)
        self.assertTrue(any("bad prints" in i for i in issues))

    def test_stale_warning(self):
        df = synthetic.daily_bars(50, start="2026-01-01")
        now = pd.Timestamp("2026-06-01 18:00", tz="Asia/Kolkata").to_pydatetime()
        r = quality.report(df, "X", now=now)
        self.assertEqual(r["status"], "STALE")
        self.assertTrue(any("stale" in w for w in r["warnings"]))


class RiskAndSim(unittest.TestCase):
    def setUp(self):
        self.rc = RiskConfig()
        self.day = pd.Timestamp("2024-01-02")

    def bar(self, o, h, l, c):
        return pd.Series({"open": o, "high": h, "low": l, "close": c})

    def test_sizing_uses_smallest_limit(self):
        rm = RiskManager(self.rc)
        d = rm.check_entry("X", self.day, 100.0, 2.0, 1_000_000, 1_000_000, 0)
        # risk: 10,000 / (2*2) = 2500 shares; cap 25% = 2500; cash = 9950 -> 2500
        self.assertTrue(d.approved)
        self.assertEqual(d.qty, 2500)
        self.assertEqual((d.stop, d.target), (96.0, 108.0))
        self.assertFalse(rm.check_entry("X", self.day, 100.0, 2.0, 1e6, 1e6, 5).approved)  # max positions
        self.assertFalse(rm.check_entry("X", self.day, 100.0, float("nan"), 1e6, 1e6, 0).approved)

    def test_cooldown_daily_loss_and_drawdown_pause(self):
        rm = RiskManager(self.rc)
        rm.on_trade_closed("X", self.day, -500)
        self.assertFalse(rm.check_entry("X", self.day + pd.offsets.BDay(2), 100, 2, 1e6, 1e6, 0).approved)
        self.assertTrue(rm.check_entry("X", self.day + pd.offsets.BDay(4), 100, 2, 1e6, 1e6, 0).approved)
        rm.on_close(self.day, 1_000_000)
        rm.on_close(self.day + pd.offsets.BDay(1), 970_000)          # -3% day
        self.assertFalse(rm.check_entry("Y", self.day + pd.offsets.BDay(2), 100, 2, 1e6, 1e6, 0).approved)
        self.assertTrue(rm.check_entry("Y", self.day + pd.offsets.BDay(3), 100, 2, 1e6, 1e6, 0).approved)
        rm.on_close(self.day + pd.offsets.BDay(3), 800_000)          # 20% drawdown -> pause
        self.assertTrue(rm.halted)
        rm.on_close(self.day + pd.offsets.BDay(30), 800_000)         # pause over, new high-water mark
        self.assertFalse(rm.halted)
        self.assertEqual(rm.peak_equity, 800_000)

    def test_next_open_fill_stop_gap_and_accounting(self):
        sim = PortfolioSim(1_000_000, RiskManager(self.rc), slippage_bps=10)
        sim.step(self.day, {"X": self.bar(100, 101, 99, 100)})
        sim.order("X", "BUY", "test", atr=2.0, signal_date=self.day)
        self.assertEqual(sim.positions, {})                         # nothing fills at the signal close
        d2 = self.day + pd.offsets.BDay(1)
        sim.step(d2, {"X": self.bar(102, 103, 101, 102)})
        p = sim.positions["X"]
        self.assertEqual(p.entry_date, "2024-01-03")
        self.assertEqual(p.entry_price, 102.15)                      # 102 * 1.001 rounded up to the tick
        d3 = d2 + pd.offsets.BDay(1)
        sim.step(d3, {"X": self.bar(90, 95, 89, 94)})                # gap below the stop
        t = sim.trades[0]
        self.assertEqual(t["exit_reason"], "stop_loss")
        self.assertLess(t["exit_price"], p.stop)                     # filled at the open, worse than the stop
        self.assertAlmostEqual(sim.cash, 1_000_000 + t["net_pnl"], places=2)
        self.assertAlmostEqual(sim.charges_paid, t["charges"], places=2)

    def test_stop_assumed_before_target_on_same_bar(self):
        sim = PortfolioSim(1_000_000, RiskManager(self.rc), slippage_bps=0)
        sim.step(self.day, {"X": self.bar(100, 100, 100, 100)})
        sim.order("X", "BUY", "t", atr=1.0, signal_date=self.day)
        sim.step(self.day + pd.offsets.BDay(1), {"X": self.bar(100, 100, 100, 100)})
        sim.step(self.day + pd.offsets.BDay(2), {"X": self.bar(100, 120, 80, 100)})
        self.assertEqual(sim.trades[0]["exit_reason"], "stop_loss")

    def test_backtest_end_to_end_and_buy_and_hold(self):
        cfg = config.load()
        data = {"A": indicators.add_all(synthetic.daily_bars(800, seed=5))}
        r = backtest.run(data, "ma_crossover", cfg)
        self.assertGreater(r["metrics"]["trades"], 3)
        self.assertAlmostEqual(r["metrics"]["final_equity"], cfg["paper"]["capital"] + sum(t["net_pnl"] for t in r["trades"]), places=0)
        for t in r["trades"]:
            self.assertGreater(t["entry_date"], r["orders"][0]["signal_date"] or "")
        bh = backtest.buy_and_hold(data, cfg)
        self.assertAlmostEqual(bh["metrics"]["final_equity"], cfg["paper"]["capital"] + bh["trades"][0]["net_pnl"], places=0)


class WalkForward(unittest.TestCase):
    def test_purged_folds_and_no_skill_on_random_walk(self):
        d = indicators.add_all(synthetic.daily_bars(1400, seed=11, regime_len=10**6, drift=0))
        X, (y, _) = features.build(d), features.labels(d, 5)
        wf = walkforward.run(X, y, "logistic", horizon=5, min_train=500, test_bars=100)
        for f in wf.folds:
            gap = len(pd.bdate_range(f["train_end"], f["test_start"])) - 1
            self.assertGreaterEqual(gap, 6)                     # horizon + 1 bars purged
        self.assertGreater(wf.pred.index[0], X.dropna().index[499])
        self.assertLess(abs(wf.summary["auc"] - 0.5), 0.07)    # random walk: no skill, no leak

    def test_leaky_feature_would_be_caught(self):
        """Sanity check of the metric: a feature equal to the future label scores near-perfect AUC."""
        d = indicators.add_all(synthetic.daily_bars(1200, seed=4))
        X, (y, _) = features.build(d), features.labels(d, 5)
        X = X.assign(leak=y.fillna(0))
        wf = walkforward.run(X, y, "logistic", horizon=5, min_train=500, test_bars=100)
        self.assertGreater(wf.summary["auc"], 0.95)


class LLMAdvisor(unittest.TestCase):
    def test_unavailable_and_malformed(self):
        cfg = config.load()
        with mock.patch("quant.llm.requests.get", side_effect=__import__("requests").ConnectionError()):
            self.assertEqual(Ollama(cfg).review({})["stance"], "unavailable")
        llm = Ollama(cfg)
        llm._ok = True
        resp = mock.Mock(ok=True)
        resp.json.return_value = {"message": {"content": '{"stance": "BUY 1000 shares now", "commentary": "x"}'}}
        with mock.patch("quant.llm.requests.post", return_value=resp):
            self.assertEqual(llm.review({})["stance"], "caution")  # unknown instructions are ignored


class AgentsAndPaper(unittest.TestCase):
    def test_full_cycle_paper_trading(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = tmp_cfg(d)
            cfg["ml"]["hybrid_threshold"] = 0.0          # make sure the synthetic run proposes trades
            store = seeded_store(cfg)
            res = pipeline.train(store, cfg)
            self.assertTrue(all("error" not in r for r in res), res)
            dates = store.bars("^NSEI").index
            acct = PaperAccount(cfg)
            seen_orders = 0
            for t in dates[-60:]:
                st = agents.run(store, cfg, acct, fetch=False, use_llm=False, end=t.strftime("%Y-%m-%d"))
                for p in st["approved"]:
                    self.assertIn(p["side"], ("BUY", "SELL"))
                seen_orders += len(st["approved"])
                for tr in acct.trades():
                    self.assertGreater(tr["entry_date"], "")
            self.assertGreater(seen_orders, 0)
            self.assertEqual(len(acct.equity_curve()), 60)
            # every fill happened at a session AFTER its signal (next-open rule)
            for o in acct.orders(1000):
                if o["side"] == "BUY" and o["status"] == "filled":
                    self.assertGreater(o["date"], o["signal_date"])
            # state survives a restart
            again = PaperAccount(cfg)
            self.assertEqual(again.sim.cash, acct.sim.cash)
            self.assertEqual(set(again.sim.positions), set(acct.sim.positions))
            self.assertIn(acct.last_agent_run()["engine"], ("langgraph", "sequential"))

    def test_llm_veto_blocks_entries_only(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = tmp_cfg(d)
            acct = PaperAccount(cfg)
            llm = mock.Mock()
            llm.review.return_value = {"stance": "veto", "commentary": "no", "source": "test"}
            agent = agents.StrategyGenerationAgent(cfg, acct, llm)
            data = {"AAA.NS": indicators.add_all(synthetic.daily_bars(300))}
            tech = {"AAA.NS": {"date": "2020-01-01", "algo_setup": True, "algo_exit": False, "atr14": 1.0, "close": 10.0}}
            st = {"as_of": "2020-01-01", "data": data, "technical": tech, "quality": {"AAA.NS": {"status": "OK"}},
                  "ml": {"AAA.NS": {"p_up": 0.9}}, "market": {}}
            self.assertEqual(agent(st)["proposals"], [])
            llm.review.return_value = {"stance": "agree", "commentary": "ok", "source": "test"}
            self.assertEqual(agent(st)["proposals"][0]["side"], "BUY")

    def test_paper_only_cannot_be_disabled(self):
        from arena.safety import RealMoneyBlocked, install_guards
        cfg = copy.deepcopy(config.load())
        cfg["safety"]["paper_only"] = False
        with self.assertRaises(RealMoneyBlocked):
            install_guards(cfg)


if __name__ == "__main__":
    unittest.main()


class LiveSessionTests(unittest.TestCase):
    """A whole simulated trading day with a fake clock and fake 1-minute quotes."""

    def test_open_fill_stop_and_close(self):
        from datetime import datetime, timedelta
        from quant.clock import IST
        from quant.live import LiveSession
        with tempfile.TemporaryDirectory() as d:
            cfg = tmp_cfg(d)
            cfg["data"]["watchlist"] = ["AAA.NS"]
            acct = PaperAccount(cfg)
            acct.queue("AAA.NS", "BUY", "test", atr=2.0, signal_date="2026-11-26")
            acct.last_date = "2026-11-26"
            day = datetime(2026, 11, 27, tzinfo=IST)
            idx = pd.date_range(day.replace(hour=9, minute=15), day.replace(hour=15, minute=29), freq="1min")
            px = np.full(len(idx), 100.0)
            px[120:] = 95.0                                   # 11:15 drop below the 96 stop
            bars = pd.DataFrame({"open": px, "high": px + 0.1, "low": px - 0.1, "close": px, "adj_close": px, "volume": 1e4}, index=idx)
            clock = {"t": day.replace(hour=9, minute=10)}

            def sleep(sec):
                clock["t"] += timedelta(seconds=sec)

            quotes = lambda syms: {"AAA.NS": bars[bars.index <= clock["t"]]}
            r = LiveSession(cfg, acct, quotes=quotes, clock=lambda: clock["t"], sleep=sleep, poll_seconds=60).run()
            self.assertEqual(r["status"], "closed")
            t = acct.trades()[0]
            self.assertEqual(t["entry_date"], "2026-11-27")
            self.assertEqual(t["entry_price"], 100.05)       # 09:15 open + 5 bps, rounded up to the tick
            self.assertEqual(t["exit_reason"], "stop_loss")
            self.assertEqual(acct.last_date, "2026-11-27")    # the evening run will not re-process today
            self.assertIn("2026-11-27", [str(x.date()) for x in acct.equity_curve().index])
            self.assertEqual(acct.live_status()["state"], "closed")
            again = LiveSession(cfg, PaperAccount(cfg), quotes=quotes, clock=lambda: clock["t"].replace(hour=10), sleep=sleep).run()
            self.assertEqual(again["status"], "already_processed")

    def test_holiday_ends_session(self):
        from datetime import datetime, timedelta
        from quant.clock import IST
        from quant.live import LiveSession
        with tempfile.TemporaryDirectory() as d:
            cfg = tmp_cfg(d)
            clock = {"t": datetime(2026, 11, 27, 9, 20, tzinfo=IST)}
            r = LiveSession(cfg, PaperAccount(cfg), quotes=lambda s: {}, clock=lambda: clock["t"],
                            sleep=lambda sec: clock.__setitem__("t", clock["t"] + timedelta(seconds=sec))).run()
            self.assertEqual(r["status"], "no_data")


class UnfinishedBar(unittest.TestCase):
    def test_todays_bar_ignored_until_close(self):
        from datetime import datetime
        from quant.clock import IST
        df = synthetic.daily_bars(5, start="2026-11-23")             # Mon..Fri 27 Nov
        during = datetime(2026, 11, 27, 13, 40, tzinfo=IST)
        after = datetime(2026, 11, 27, 15, 35, tzinfo=IST)
        self.assertEqual(str(quality.drop_unfinished_session(df, during).index[-1].date()), "2026-11-26")
        self.assertEqual(str(quality.drop_unfinished_session(df, after).index[-1].date()), "2026-11-27")
