"""Streamlit dashboard: python -m quant dashboard   (paper trading only)

Every chart and table states its data source, the newest bar's timestamp, when it was
fetched, and whether it is historical, delayed or missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from arena.safety import install_guards
from quant import config, metrics, pipeline
from quant.data import quality
from quant.data.store import Store
from quant.paper import PaperAccount

st.set_page_config(page_title="AI Trading Agent (paper)", page_icon="📈", layout="wide")
CFG = config.load()
install_guards(CFG)
UP, DOWN, ACCENT = "#16a34a", "#dc2626", "#2563eb"


@st.cache_resource
def store() -> Store:
    return Store(CFG["data"]["db"])


@st.cache_data(ttl=60, show_spinner=False)
def prepared(sym: str, fetched_at: str | None) -> pd.DataFrame:
    return pipeline.prepared(store(), sym)


def meta_line(rep: dict) -> str:
    if rep.get("rows", 0) == 0:
        return "No data stored."
    return (f"Source: **{', '.join(rep.get('sources') or ['?'])}** · last bar **{rep['last'][:10]}** · "
            f"fetched {str(rep.get('fetched_at'))[:16].replace('T', ' ')} IST · {rep['kind']}")


def warn(rep: dict) -> None:
    if rep.get("status") in ("STALE", "MISSING"):
        st.error(f"{rep['symbol']}: data {rep['status']}. " + " ".join(rep["warnings"][:2]))
    for w in rep.get("warnings", []):
        if rep.get("status") == "OK":
            st.caption(f"⚠ {w}")


def rs(x: float) -> str:
    return ("−" if x < 0 else "") + f"₹{abs(x):,.0f}"


s = store()
reports = {sym: quality.report(s.bars(sym), sym, stale_after_hours=CFG["data"]["stale_after_hours"]) for sym in pipeline.all_symbols(CFG)}

# ---- sidebar --------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 AI + Algo Trading Agent")
    st.warning("PAPER TRADING ONLY · virtual money · not investment advice", icon="⚠️")
    st.caption("Free data: Yahoo Finance (unofficial, end-of-day). Not exchange-licensed; may be late or wrong.")
    if st.button("Fetch latest data", width="stretch"):
        with st.spinner("Downloading from Yahoo Finance..."):
            res = pipeline.fetch(s, CFG)
        bad = [r["symbol"] for r in res if r["status"] != "ok"]
        (st.error(f"Failed: {bad}") if bad else st.success("Data updated."))
        st.cache_data.clear()
    if st.button("Run agent cycle (paper)", width="stretch", type="primary"):
        from quant import agents
        with st.spinner("Running the 9 agents..."):
            stt = agents.run(s, CFG, PaperAccount(CFG), fetch=False)
        st.success(f"Agent cycle done for {stt['as_of']}.")
    st.divider()
    st.caption("Run from a terminal:\n\n`python -m quant autopilot` (whole day)\n\n`python -m quant live` (market hours)\n\n"
               "`python -m quant daily` (after 15:30)\n\n`python -m quant compare`")

tabs = st.tabs(["Market overview", "Charts & indicators", "Signals & AI", "Agent analysis", "Paper portfolio",
                "Backtests", "Risk"])

# ---- 1. market overview --------------------------------------------------
with tabs[0]:
    st.subheader("Market overview")
    cols = st.columns(len(CFG["data"]["indices"]))
    for col, sym in zip(cols, CFG["data"]["indices"]):
        rep = reports[sym]
        d = prepared(sym, rep.get("fetched_at"))
        with col:
            if d.empty:
                st.error(f"{sym}: no data. Run fetch.")
                continue
            chg = d["close"].iloc[-1] / d["close"].iloc[-2] - 1
            name = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY"}.get(sym, sym)
            st.metric(name, f"{d['close'].iloc[-1]:,.2f}", f"{chg:+.2%}")
            st.caption(meta_line(rep))
            warn(rep)
            fig = go.Figure(go.Scatter(x=d.index[-260:], y=d["close"].iloc[-260:], line=dict(color=ACCENT, width=2), name=name))
            fig.add_scatter(x=d.index[-260:], y=d["ema200"].iloc[-260:], line=dict(color="#9ca3af", dash="dot"), name="EMA200")
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, width="stretch")
    st.markdown("#### Data quality")
    st.dataframe(pd.DataFrame([{"symbol": k, "status": v["status"], "rows": v["rows"], "last bar": v.get("last", "")[:10],
                                "kind": v.get("kind", ""), "source": ", ".join(v.get("sources") or []),
                                "fetched (IST)": str(v.get("fetched_at", ""))[:16], "warnings": " | ".join(v["warnings"])}
                               for k, v in reports.items()]), width="stretch", hide_index=True)
    with st.expander("Limits of free data"):
        st.markdown(
            "- **Yahoo Finance** (via yfinance) is unofficial: bars can be missing, late or briefly wrong. The cleaner drops "
            "obvious bad prints and reports them; it never invents prices.\n"
            "- **End-of-day only** for strategy decisions. Intraday history is short (1-minute: ~7 days; 5-15-minute: 60 days; hourly: ~2 years) and quotes are delayed.\n"
            "- **Index tickers** carry no reliable volume, so volume filters skip them. Indices cannot be bought; NIFTYBEES/BANKBEES ETFs stand in.\n"
            "- **No free history** of options chains, tick data, or point-in-time index constituents (survivorship bias).\n"
            "- NSE holidays are not available as free machine-readable data; a missing bar on a weekday may be a holiday.")

# ---- 2. charts --------------------------------------------------------------
with tabs[1]:
    c1, c2 = st.columns([2, 1])
    sym = c1.selectbox("Symbol", pipeline.all_symbols(CFG), key="chart_sym")
    span = c2.selectbox("Period", ["6M", "1Y", "2Y", "5Y", "All"], index=1)
    rep = reports[sym]
    d = prepared(sym, rep.get("fetched_at"))
    st.caption(meta_line(rep) + " · prices adjusted for splits and dividends")
    warn(rep)
    if not d.empty:
        n = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": len(d)}[span]
        v = d.iloc[-n:]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.52, 0.16, 0.16, 0.16], vertical_spacing=0.02,
                            subplot_titles=("", "RSI 14", "MACD 12/26/9", "ADX 14 with +DI / −DI"))
        fig.add_candlestick(x=v.index, open=v.open, high=v.high, low=v.low, close=v.close, name="price",
                            increasing_line_color=UP, decreasing_line_color=DOWN, row=1, col=1)
        for c, col in (("ema21", "#f59e0b"), ("ema50", ACCENT), ("ema200", "#6b7280")):
            fig.add_scatter(x=v.index, y=v[c], name=c.upper(), line=dict(color=col, width=1.4), row=1, col=1)
        fig.add_scatter(x=v.index, y=v.bb_up, name="BB upper", line=dict(color="#a78bfa", width=1, dash="dot"), row=1, col=1)
        fig.add_scatter(x=v.index, y=v.bb_lo, name="BB lower", line=dict(color="#a78bfa", width=1, dash="dot"), row=1, col=1)
        if v["vwap"].notna().any():
            fig.add_scatter(x=v.index, y=v.vwap, name="VWAP 20d", line=dict(color="#14b8a6", width=1), row=1, col=1)
        fig.add_scatter(x=v.index, y=v.rsi14, name="RSI", line=dict(color=ACCENT), row=2, col=1)
        for lvl in (30, 70):
            fig.add_hline(y=lvl, line=dict(color="#9ca3af", dash="dot"), row=2, col=1)
        fig.add_bar(x=v.index, y=v.macd_hist, name="MACD hist", marker_color=[UP if x >= 0 else DOWN for x in v.macd_hist], row=3, col=1)
        fig.add_scatter(x=v.index, y=v.macd, name="MACD", line=dict(color=ACCENT, width=1.2), row=3, col=1)
        fig.add_scatter(x=v.index, y=v.macd_signal, name="Signal", line=dict(color="#f59e0b", width=1.2), row=3, col=1)
        fig.add_scatter(x=v.index, y=v.adx, name="ADX", line=dict(color="#111827"), row=4, col=1)
        fig.add_scatter(x=v.index, y=v.plus_di, name="+DI", line=dict(color=UP, width=1), row=4, col=1)
        fig.add_scatter(x=v.index, y=v.minus_di, name="−DI", line=dict(color=DOWN, width=1), row=4, col=1)
        fig.update_layout(height=900, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=90, b=0),
                          legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left"))
        st.plotly_chart(fig, width="stretch")
        last = v.iloc[-1]
        st.dataframe(pd.DataFrame([{"date": v.index[-1].date(), "close": last.close, "RSI": last.rsi14, "MACD hist": last.macd_hist,
                                    "ADX": last.adx, "ATR %": last.atr_pct, "%B": last.bb_pctb, "vol x avg": last.vol_ratio}]).round(2),
                     hide_index=True, width="stretch")

# ---- 3. signals and AI ------------------------------------------------------
live = PaperAccount(CFG)
run = live.last_agent_run()
with tabs[2]:
    st.subheader("Signals and ML predictions")
    if not run:
        st.info("No agent cycle yet. Click **Run agent cycle** or run `python -m quant paper run`.")
    else:
        st.caption(f"From the agent cycle at {run['ts'][:16].replace('T', ' ')} IST, using end-of-day data for {run['as_of']}.")
        rows = []
        for k, t in run.get("technical", {}).items():
            m = run.get("ml", {}).get(k, {})
            rows.append({"symbol": k, "bar date": t["date"], "close": t["close"], "RSI": t["rsi14"], "ADX": t["adx"],
                         "algo setup": "✅" if t["algo_setup"] else "", "algo exit": "⛔" if t["algo_exit"] else "",
                         **{f"{n}": v for n, v in t["signals"].items()},
                         "ML P(up 5d)": m.get("p_up"), "model AUC (walk-fwd)": m.get("walk_forward_auc"),
                         "trained through": m.get("trained_through")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("P(up) = model's probability that the next-5-session return beats ~0.3% costs. A walk-forward AUC near 0.50 means "
                   "the model has shown no real out-of-sample skill; treat its probability as weak evidence.")
    psym = st.selectbox("Walk-forward predictions for", CFG["data"]["watchlist"])
    p = s.predictions(psym, CFG["ml"]["model"])
    d = prepared(psym, reports[psym].get("fetched_at"))
    if len(p) and not d.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=d.index[d.index >= p.index[0]], y=d["close"][d.index >= p.index[0]], name="close", line=dict(color="#6b7280"))
        fig.add_scatter(x=p.index, y=p.rolling(5).mean(), name="P(up), 5-day mean", line=dict(color=ACCENT), secondary_y=True)
        fig.add_hline(y=0.5, line=dict(dash="dot", color="#9ca3af"), secondary_y=True)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
        _, meta = s.load_model(f"{psym}:{CFG['ml']['model']}")
        if meta:
            st.json({"walk_forward_out_of_sample": meta["walk_forward"], "top_features": meta.get("importances")}, expanded=False)
    else:
        st.info("No walk-forward predictions yet. Run `python -m quant train`.")

# ---- 4. agent analysis ------------------------------------------------------
with tabs[3]:
    st.subheader("Agent analysis")
    if run:
        st.caption(f"Orchestration: {run.get('engine')} · run at {run['ts'][:16].replace('T', ' ')} IST · data as of {run['as_of']}")
        for e in run["log"]:
            with st.container(border=True):
                st.markdown(f"**{e['agent']}** — {e['summary']}")
                if e.get("details"):
                    st.json(e["details"], expanded=False)
        c1, c2 = st.columns(2)
        c1.markdown("**Approved by risk manager**")
        c1.dataframe(pd.DataFrame(run.get("approved", [])), width="stretch", hide_index=True)
        c2.markdown("**Rejected by risk manager**")
        c2.dataframe(pd.DataFrame(run.get("rejected", [])), width="stretch", hide_index=True)
        llm = [p for p in run.get("proposals", []) if p.get("llm")]
        if llm:
            st.markdown("**Local LLM commentary (advisory only; it can veto a new entry, nothing else)**")
            st.dataframe(pd.DataFrame([{"symbol": p["symbol"], **p["llm"]} for p in llm]), hide_index=True, width="stretch")
    else:
        st.info("No agent cycle yet.")

# ---- 5. paper portfolio -----------------------------------------------------
with tabs[4]:
    replay_path = Path(CFG["paper"]["db"]).with_name("paper_replay.db")
    which = st.radio("Account", ["Live paper account", "Paper replay (historical forward test)"], horizontal=True)
    acct = live if which.startswith("Live") or not replay_path.exists() else PaperAccount(CFG, str(replay_path))
    if which.startswith("Paper replay"):
        st.info("Replay: the agents traded past sessions one day at a time, with models trained only on data before the replay "
                "window. It is a historical simulation, not live trading.")
    ls = acct.live_status() if acct is live else None
    if ls:
        label = {"running": "🟢 LIVE session running", "closed": "Session closed", "no_data": "No live data (holiday?)"}.get(ls["state"], ls["state"])
        st.markdown(f"**{label}** · last update {ls['updated_at'][11:16]} IST on {ls['updated_at'][:10]} · {ls['source']}")
        if ls.get("prices"):
            st.dataframe(pd.DataFrame([{"symbol": k, "price (delayed)": v["price"], "1-min bar": v["bar_time"],
                                        "held": "✅" if k in acct.sim.positions else ""} for k, v in ls["prices"].items()]),
                         hide_index=True, width="stretch")
        st.caption("Live prices are free Yahoo 1-minute bars: unofficial and delayed. Stops are checked every minute; "
                   "new BUY/SELL decisions are made after the close.")
    sim, eq, tr = acct.sim, acct.equity_curve(), acct.trades()
    m = metrics.compute(eq, tr, sim.capital, sim.charges_paid, sim.slippage_paid) if len(eq) else {}
    c = st.columns(5)
    c[0].metric("Equity (virtual)", rs(sim.total_equity), f"{(sim.total_equity / sim.capital - 1):+.2%}")
    c[1].metric("Cash", rs(sim.cash))
    c[2].metric("Open positions", len(sim.positions))
    c[3].metric("Closed trades", m.get("trades", 0), f"win rate {m.get('win_rate_pct', 0)}%")
    c[4].metric("Charges + slippage", rs(sim.charges_paid + sim.slippage_paid))
    st.caption(f"As of {acct.last_date} (end-of-day). Fills happen at the next session's open with slippage and Indian charges.")
    if len(eq) > 1:
        fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, fill="tozeroy", line=dict(color=ACCENT), name="equity"))
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(range=[eq.min() * 0.98, eq.max() * 1.01]))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Open positions**")
    st.dataframe(pd.DataFrame([{**vars(p), "unrealized": round((p.last_price - p.entry_price) * p.qty, 2)} for p in sim.positions.values()]),
                 width="stretch", hide_index=True)
    if sim.pending:
        st.markdown("**Orders waiting for the next open**")
        st.dataframe(pd.DataFrame([{"symbol": k, **v} for k, v in sim.pending.items()]), hide_index=True, width="stretch")
    st.markdown("**Trade history**")
    st.dataframe(pd.DataFrame(tr[::-1]), width="stretch", hide_index=True)
    with st.expander("Event log"):
        st.dataframe(pd.DataFrame(acct.events(300)), width="stretch", hide_index=True)

# ---- 6. backtests -----------------------------------------------------------
with tabs[5]:
    st.subheader("Backtest comparison")
    bts = [b for b in s.latest_backtests() if b["symbol"] == "PORTFOLIO"]
    if not bts:
        st.info("No backtests yet. Run `python -m quant train` then `python -m quant compare`.")
    else:
        rows = [{"strategy": b["strategy"], **json.loads(b["metrics_json"])} for b in bts]
        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
        cfgj = json.loads(bts[0]["config_json"])
        st.caption(f"Window {bts[0]['start']} to {bts[0]['end']} (walk-forward out-of-sample) · symbols {', '.join(cfgj.get('symbols', []))} · "
                   f"ML model {cfgj.get('model')} · run {bts[0]['run_at'][:16]}")
        show = ["strategy", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "trades", "win_rate_pct",
                "profit_factor", "avg_win", "avg_loss", "charges_paid", "slippage_paid", "exposure_pct"]
        st.dataframe(df[show], width="stretch", hide_index=True)
        pick = st.multiselect("Equity curves", df["strategy"].tolist(), default=df["strategy"].tolist()[:4])
        fig, fdd = go.Figure(), go.Figure()
        for b in bts:
            if b["strategy"] in pick:
                e = pd.Series(json.loads(b["equity_json"]))
                e.index = pd.to_datetime(e.index)
                fig.add_scatter(x=e.index, y=e.values, name=b["strategy"])
                fdd.add_scatter(x=e.index, y=metrics.drawdown(e).values * 100, name=b["strategy"])
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="equity ₹")
        fdd.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="drawdown %")
        st.plotly_chart(fig, width="stretch")
        st.plotly_chart(fdd, width="stretch")
        st.warning("Simulated results with modelled costs. They do not show future profitability. Survivorship bias: the "
                   "watchlist is today's large caps, chosen with hindsight. LLMs are excluded from backtests (look-ahead).")

# ---- 7. risk ----------------------------------------------------------------
with tabs[6]:
    st.subheader("Risk")
    r = live.sim.risk
    eq = live.equity_curve()
    dd = max(0.0, float(metrics.drawdown(eq).iloc[-1] * -100)) if len(eq) else 0.0
    c = st.columns(4)
    c[0].metric("Current drawdown", f"{dd:.2f}%", f"limit {r.cfg.max_drawdown_pct}%", delta_color="off")
    c[1].metric("New entries", "PAUSED" if r.halted else "allowed", r.halt_until or "")
    c[2].metric("Daily-loss block", r.blocked_until or "none")
    exposure = live.sim.holdings / live.sim.total_equity * 100 if live.sim.total_equity else 0
    c[3].metric("Exposure", f"{exposure:.1f}%")
    st.markdown("**Limits (enforced in code; agents and the LLM cannot change them)**")
    st.dataframe(pd.DataFrame([r.to_state()["cfg"]]).T.rename(columns={0: "value"}), width="stretch")
    if r.cooldown:
        st.markdown("**Cooldowns after losses**")
        st.json(r.cooldown)
