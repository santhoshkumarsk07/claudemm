"""The nine agents and their orchestration (LangGraph, with a plain fallback).

    data_collection -> paper_sync -> market_trend -> technical -> ml_prediction
      -> strategy -> risk -> paper_execute -> backtesting -> performance

Responsibilities are separate and so is authority:
- Analysis agents (trend, technical, ML, LLM commentary) only describe.
- The strategy agent turns fixed, backtested rules into PROPOSALS.
- The risk agent (deterministic code) approves, sizes or rejects each proposal.
- Only the paper-trading agent changes the virtual account, and only with approved orders.
- The optional LLM can veto a new entry; it cannot create, enlarge or un-reject anything.
"""
from __future__ import annotations

import copy
import logging
import operator
from typing import Annotated, Any, TypedDict

import pandas as pd

from . import metrics, pipeline, strategies
from .data import quality
from .data.store import Store
from .llm import Ollama
from .paper import PaperAccount

log = logging.getLogger("quant")


class AgentState(TypedDict, total=False):
    as_of: str
    data: dict
    quality: dict
    fetch: list
    market: dict
    technical: dict
    ml: dict
    proposals: list
    approved: list
    rejected: list
    paper: dict
    backtest: dict
    performance: dict
    log: Annotated[list, operator.add]


def _note(agent: str, text: str, **details) -> dict:
    return {"log": [{"agent": agent, "summary": text, **({"details": details} if details else {})}]}


# ---- 1. data collection --------------------------------------------------
class DataCollectionAgent:
    name = "Data Collection"

    def __init__(self, store: Store, cfg: dict, fetch: bool = True, end: str | None = None):
        self.store, self.cfg, self.do_fetch, self.end = store, cfg, fetch, end

    def __call__(self, s: AgentState) -> dict:
        fetched = pipeline.fetch(self.store, self.cfg) if self.do_fetch else []
        data, rep = {}, {}
        now = None
        if self.end:  # replay: pretend it is the evening of `end`; nothing later is visible
            from .clock import IST
            now = pd.Timestamp(self.end).to_pydatetime().replace(hour=18, tzinfo=IST)
        for sym in pipeline.all_symbols(self.cfg):
            d = pipeline.prepared(self.store, sym, self.end)
            rep[sym] = quality.report(self.store.bars(sym, end=self.end), sym, "1d", now=now,
                                      stale_after_hours=self.cfg["data"]["stale_after_hours"])
            if not d.empty:
                data[sym] = d
        if not data:
            raise RuntimeError("No market data stored. Run `python -m quant fetch` (needs internet access to Yahoo Finance).")
        as_of = max(d.index[-1] for d in data.values()).strftime("%Y-%m-%d")
        errs = [f["symbol"] for f in fetched if f["status"] != "ok"]
        stale = [k for k, r in rep.items() if r["status"] != "OK"]
        msg = f"{len(data)} symbols loaded, latest session {as_of}. Source: Yahoo Finance (end-of-day, unofficial)."
        if errs:
            msg += f" Fetch failed for {errs} (using stored data)."
        if stale:
            msg += f" Not current: {stale}."
        return {"as_of": as_of, "data": data, "quality": rep, "fetch": fetched, **_note(self.name, msg)}


# ---- 2. market trend -------------------------------------------------------
class MarketTrendAgent:
    name = "Market Trend Analysis"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @staticmethod
    def regime(d: pd.DataFrame) -> dict:
        r = d.iloc[-1]
        if r.close > r.ema200 and r.ema50 > r.ema200:
            label = "bullish trend" if r.adx >= 20 else "bullish, weak trend"
        elif r.close < r.ema200 and r.ema50 < r.ema200:
            label = "bearish trend" if r.adx >= 20 else "bearish, weak trend"
        else:
            label = "mixed / sideways"
        return {"date": str(d.index[-1].date()), "close": round(r.close, 2), "regime": label, "adx": round(r.adx, 1),
                "rsi": round(r.rsi14, 1), "ret_20d_pct": round((r.close / d["close"].iloc[-21] - 1) * 100, 2),
                "vs_ema200_pct": round((r.close / r.ema200 - 1) * 100, 2)}

    def __call__(self, s: AgentState) -> dict:
        idx = {k: self.regime(s["data"][k]) for k in self.cfg["data"]["indices"] if k in s["data"]}
        wl = [k for k in self.cfg["data"]["watchlist"] if k in s["data"]]
        breadth = sum(s["data"][k].iloc[-1].close > s["data"][k].iloc[-1].ema50 for k in wl) / max(len(wl), 1) * 100
        out = {"indices": idx, "breadth_above_ema50_pct": round(breadth, 1)}
        txt = "; ".join(f"{k}: {v['regime']} (ADX {v['adx']}, {v['ret_20d_pct']:+.1f}% in 20d)" for k, v in idx.items())
        return {"market": out, **_note(self.name, f"{txt}. {breadth:.0f}% of the watchlist is above its 50-day EMA.")}


# ---- 3. technical analysis -------------------------------------------------
class TechnicalAnalysisAgent:
    name = "Technical Analysis"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def __call__(self, s: AgentState) -> dict:
        tech = {}
        for sym in self.cfg["data"]["watchlist"]:
            d = s["data"].get(sym)
            if d is None or len(d) < 210:
                continue
            r = d.iloc[-1]
            sig = {}
            for name in strategies.TRADITIONAL:
                e, x = strategies.signals(name, d)
                sig[name] = "entry" if e.iloc[-1] else "exit" if x.iloc[-1] else "-"
            tech[sym] = {"date": str(d.index[-1].date()), "close": round(r.close, 2), "rsi14": round(r.rsi14, 1),
                         "macd_hist": round(r.macd_hist, 3), "adx": round(r.adx, 1), "atr_pct": round(r.atr_pct, 2),
                         "ema21_gt_ema50": bool(r.ema21 > r.ema50), "above_ema200": bool(r.close > r.ema200),
                         "bb_pctb": round(r.bb_pctb, 2), "vol_ratio": None if pd.isna(r.vol_ratio) else round(r.vol_ratio, 2),
                         "algo_setup": bool(strategies.algo_setup(d).iloc[-1]),
                         "algo_exit": bool(strategies.algo_exit(d).iloc[-1]), "signals": sig, "atr14": float(r.atr14)}
        n_setup = sum(v["algo_setup"] for v in tech.values())
        return {"technical": tech, **_note(self.name, f"{n_setup} of {len(tech)} stocks have an active algorithmic long setup.")}


# ---- 4. machine-learning prediction ----------------------------------------
class MLPredictionAgent:
    name = "ML Prediction"

    def __init__(self, store: Store, cfg: dict, prefix: str = ""):
        self.store, self.cfg, self.prefix = store, cfg, prefix
        self.model = cfg["ml"]["model"]

    def series(self, sym: str, d: pd.DataFrame, market: pd.DataFrame | None):
        """P(up) for every bar from the saved final model (in-sample up to its training date)."""
        from .ml import features
        m, meta = self.store.load_model(f"{self.prefix}{sym}:{self.model}")
        if m is None:
            return None, None
        X = features.build(d, market)[meta["features"]].dropna()
        return pd.Series(m.predict_proba(X.values)[:, 1], index=X.index), meta

    def __call__(self, s: AgentState) -> dict:
        out, mkt = {}, s["data"].get(pipeline.MARKET)
        for sym in s.get("technical", {}):
            p, meta = self.series(sym, s["data"][sym], mkt)
            if p is None or p.empty:
                out[sym] = {"p_up": None, "note": "no trained model (run `python -m quant train`)"}
                continue
            last = p.index[-1]
            wf = meta.get("walk_forward", {})
            out[sym] = {"p_up": round(float(p.iloc[-1]), 3), "date": str(last.date()), "model": self.model,
                        "horizon_days": meta["horizon"], "trained_through": meta["trained_through"],
                        "out_of_sample": str(last.date()) > meta["trained_through"],
                        "walk_forward_auc": wf.get("auc"), "walk_forward_accuracy": wf.get("accuracy"),
                        "base_rate": wf.get("base_rate")}
        ps = [v["p_up"] for v in out.values() if v.get("p_up") is not None]
        txt = (f"{self.model}: mean P(up over {self.cfg['ml']['horizon']}d) {sum(ps) / len(ps):.2f} across {len(ps)} stocks."
               if ps else "No trained models yet.")
        return {"ml": out, **_note(self.name, txt)}


# ---- 5. strategy generation ------------------------------------------------
class StrategyGenerationAgent:
    """Applies the backtested hybrid rule (algo setup + ML agreement) to the latest bar."""
    name = "Strategy Generation"

    def __init__(self, cfg: dict, account: PaperAccount, llm: Ollama | None):
        self.cfg, self.acct, self.llm = cfg, account, llm

    def __call__(self, s: AgentState) -> dict:
        th = self.cfg["ml"]["hybrid_threshold"]
        props, held = [], self.acct.sim.positions
        for sym, t in s.get("technical", {}).items():
            q = s["quality"].get(sym, {})
            p = s.get("ml", {}).get(sym, {}).get("p_up")
            if t["date"] != s["as_of"] or q.get("status") != "OK":
                continue  # never act on a stale bar
            if sym in held:
                if t["algo_exit"] or (p is not None and p < th - 0.1):
                    why = "algo exit" if t["algo_exit"] else f"ML P(up) {p:.2f} fell below {th - 0.1:.2f}"
                    props.append({"symbol": sym, "side": "SELL", "reason": f"hybrid exit: {why}", "p_up": p})
            elif t["algo_setup"] and p is not None and p >= th:
                props.append({"symbol": sym, "side": "BUY", "p_up": p, "atr": t["atr14"], "close": t["close"],
                              "reason": f"hybrid entry: algo setup + ML P(up) {p:.2f} >= {th}. "
                                        + strategies.explain("algo", s["data"][sym])})
        vetoed = []
        if self.llm is not None:
            for pr in props:
                if pr["side"] != "BUY":
                    continue
                pr["llm"] = self.llm.review({"proposal": pr, "market": s.get("market"), "technical": s["technical"][pr["symbol"]],
                                             "ml": s["ml"].get(pr["symbol"])})
                if pr["llm"]["stance"] == "veto":
                    vetoed.append(pr["symbol"])
        props = [p for p in props if p["symbol"] not in vetoed or p["side"] == "SELL"]
        txt = f"{sum(p['side'] == 'BUY' for p in props)} BUY and {sum(p['side'] == 'SELL' for p in props)} SELL proposals."
        if vetoed:
            txt += f" LLM vetoed new entries in {vetoed}."
        return {"proposals": props, **_note(self.name, txt)}


# ---- 6. risk management -----------------------------------------------------
class RiskManagementAgent:
    name = "Risk Management"

    def __init__(self, account: PaperAccount):
        self.acct = account

    def __call__(self, s: AgentState) -> dict:
        sim = self.acct.sim
        rm = copy.deepcopy(sim.risk)  # preview only; the binding check repeats at the fill
        ok, bad, n_open = [], [], len(sim.positions)
        for p in s.get("proposals", []):
            if p["side"] == "SELL":
                ok.append({**p, "risk": "exits are always allowed"})
                continue
            dec = rm.check_entry(p["symbol"], pd.Timestamp(s["as_of"]) + pd.offsets.BDay(1), p["close"], p["atr"],
                                 sim.total_equity, sim.cash, n_open)
            (ok if dec.approved else bad).append({**p, "risk": dec.text(), "qty_preview": dec.qty,
                                                  "stop_preview": dec.stop, "target_preview": dec.target})
            if dec.approved:
                n_open += 1
        eq = self.acct.equity_curve()
        dd = max(0.0, float(metrics.drawdown(eq).iloc[-1] * -100)) if len(eq) else 0.0
        status = {"equity": round(sim.total_equity, 2), "cash": round(sim.cash, 2), "open_positions": len(sim.positions),
                  "drawdown_pct": round(dd, 2), "halted": sim.risk.halted, "blocked_until": sim.risk.blocked_until,
                  "cooldowns": sim.risk.cooldown, "limits": sim.risk.to_state()["cfg"]}
        txt = f"{len(ok)} approved, {len(bad)} rejected. Drawdown {dd:.2f}%, {'HALTED' if sim.risk.halted else 'trading allowed'}."
        return {"approved": ok, "rejected": bad, **_note(self.name, txt, status=status)}


# ---- 7 & 8. paper trading ---------------------------------------------------
class PaperTradingAgent:
    name = "Paper Trading"

    def __init__(self, cfg: dict, account: PaperAccount, ml_agent: MLPredictionAgent):
        self.cfg, self.acct, self.ml = cfg, account, ml_agent

    def sync(self, s: AgentState) -> dict:
        """Process every session since the last run: fills at the open, stops/targets, mark-to-market.
        Missed days get the same deterministic hybrid signals the backtest uses (no LLM)."""
        data, wl = s["data"], [w for w in self.cfg["data"]["watchlist"] if w in s["data"]]
        last = self.acct.last_date
        dates = sorted({t for w in wl for t in data[w].index if (last is None or t.strftime("%Y-%m-%d") > last)
                        and t.strftime("%Y-%m-%d") <= s["as_of"]})
        if last is None:
            dates = dates[-1:]  # a new account starts today; it does not trade the past
        sig = {}
        if len(dates) > 1:
            mkt = data.get(pipeline.MARKET)
            for w in wl:
                p, _ = self.ml.series(w, data[w], mkt)
                if p is not None:
                    sig[w] = strategies.signals("hybrid", data[w], p, self.cfg)
        ev = []
        for t in dates:
            ev += self.acct.process_day(t, {w: data[w].loc[t] for w in wl if t in data[w].index})
            if t != dates[-1]:
                for w, (e, x) in sig.items():
                    if t not in data[w].index:
                        continue
                    if w in self.acct.sim.positions and x.get(t, False):
                        self.acct.queue(w, "SELL", "hybrid exit (catch-up)", None, t)
                    elif w not in self.acct.sim.positions and e.get(t, False):
                        self.acct.queue(w, "BUY", "hybrid entry (catch-up)", float(data[w].at[t, "atr14"]), t)
        txt = f"Processed {len(dates)} session(s) up to {s['as_of']}: " + ("; ".join(ev) if ev else "no fills or exits.")
        return _note(self.name, txt)

    def execute(self, s: AgentState) -> dict:
        for p in s.get("approved", []):
            self.acct.queue(p["symbol"], p["side"], p["reason"], p.get("atr"), s["as_of"])
        sim = self.acct.sim
        pos = [{"symbol": k, "qty": v.qty, "entry": v.entry_price, "entry_date": v.entry_date, "stop": v.stop,
                "target": v.target, "last": v.last_price, "unrealized": round((v.last_price - v.entry_price) * v.qty, 2)}
               for k, v in sim.positions.items()]
        paper = {"equity": round(sim.total_equity, 2), "cash": round(sim.cash, 2), "positions": pos,
                 "pending": sim.pending, "capital": sim.capital}
        txt = f"Queued {len(s.get('approved', []))} order(s) for the next open. Equity Rs {sim.total_equity:,.2f}, {len(pos)} open position(s)."
        return {"paper": paper, **_note(self.name, txt)}


# ---- 7. backtesting ---------------------------------------------------------
class BacktestingAgent:
    name = "Backtesting"

    def __init__(self, store: Store, cfg: dict, enabled: bool = False):
        self.store, self.cfg, self.enabled = store, cfg, enabled

    def __call__(self, s: AgentState) -> dict:
        if not self.enabled:
            rows = self.store.latest_backtests()
            summary = {r["strategy"]: __import__("json").loads(r["metrics_json"]) for r in rows if r["symbol"] == "PORTFOLIO"}
            return {"backtest": summary, **_note(self.name, f"Using {len(summary)} stored backtest results (run `python -m quant compare` to refresh).")}
        res = pipeline.compare(self.store, self.cfg)
        summary = {k: v["metrics"] for k, v in res["results"].items()}
        return {"backtest": summary, **_note(self.name, f"Re-ran {len(summary)} strategies on {res['start']} to {res['end']}.")}


# ---- 9. performance evaluation ---------------------------------------------
class PerformanceEvaluationAgent:
    name = "Performance Evaluation"

    def __init__(self, account: PaperAccount):
        self.acct = account

    def __call__(self, s: AgentState) -> dict:
        eq, tr = self.acct.equity_curve(), self.acct.trades()
        m = metrics.compute(eq, tr, self.acct.sim.capital, self.acct.sim.charges_paid, self.acct.sim.slippage_paid) if len(eq) else {}
        bt = s.get("backtest", {}).get("hybrid")
        txt = (f"Paper: {m.get('total_return_pct', 0):+.2f}% over {len(eq)} session(s), {m.get('trades', 0)} closed trades. "
               + (f"Backtested hybrid: CAGR {bt['cagr_pct']}%, max DD {bt['max_drawdown_pct']}%." if bt else ""))
        if len(eq) < 60:
            txt += " Too few sessions for meaningful paper statistics."
        return {"performance": m, **_note(self.name, txt)}


# ---- orchestration ---------------------------------------------------------
def build(store: Store, cfg: dict, account: PaperAccount, fetch: bool = True, use_llm: bool = True,
          run_backtest: bool = False, end: str | None = None, model_prefix: str = ""):
    ml = MLPredictionAgent(store, cfg, model_prefix)
    paper = PaperTradingAgent(cfg, account, ml)
    llm = Ollama(cfg) if use_llm and cfg["llm"]["enabled"] else None
    return [
        ("data_collection", DataCollectionAgent(store, cfg, fetch, end)),
        ("paper_sync", paper.sync),
        ("market_trend", MarketTrendAgent(cfg)),
        ("technical_analysis", TechnicalAnalysisAgent(cfg)),
        ("ml_prediction", ml),
        ("strategy_generation", StrategyGenerationAgent(cfg, account, llm)),
        ("risk_management", RiskManagementAgent(account)),
        ("paper_execute", paper.execute),
        ("backtesting", BacktestingAgent(store, cfg, run_backtest)),
        ("performance_evaluation", PerformanceEvaluationAgent(account)),
    ]


def run(store: Store, cfg: dict, account: PaperAccount, **kw) -> dict[str, Any]:
    nodes = build(store, cfg, account, **kw)
    try:
        from langgraph.graph import END, START, StateGraph
        g = StateGraph(AgentState)
        for name, fn in nodes:
            g.add_node(name, fn)
        g.add_edge(START, nodes[0][0])
        for (a, _), (b, _) in zip(nodes, nodes[1:]):
            g.add_edge(a, b)
        g.add_edge(nodes[-1][0], END)
        state = g.compile().invoke({"log": []})
        engine = "langgraph"
    except ImportError:
        state, engine = {"log": []}, "sequential"
        for _, fn in nodes:
            upd = fn(state)
            state["log"] = state["log"] + upd.pop("log", [])
            state.update(upd)
    report = {k: v for k, v in state.items() if k not in ("data",)}
    report["engine"] = engine
    account.save_agent_run(state["as_of"], report)
    return state
