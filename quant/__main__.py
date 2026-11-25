"""Command line: python -m quant <command>   (paper trading only)

  check                      Phase 1: verify Python, packages, safety guard, database, data access, Ollama
  fetch [--interval 1d]      download free Yahoo Finance bars into SQLite
  import-csv FILE SYMBOL     load a legally downloaded CSV (e.g. Kaggle) as SYMBOL
  quality                    data-quality report (gaps, stale bars, missing volume)
  indicators SYMBOL          latest indicator values for one symbol
  backtest --strategy NAME   backtest one strategy on the watchlist
  train [--models a,b]       walk-forward ML training (+ final model for paper trading)
  compare                    traditional vs ML vs hybrid on the same out-of-sample window
  paper run|status|reset     daily agent cycle on the live paper account
  paper replay --start DATE  forward-test the full agent loop on past sessions it has never seen
  dashboard                  Streamlit dashboard
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from arena.safety import install_guards

from . import config, pipeline
from .data.store import Store

log = logging.getLogger("quant")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("yfinance", "urllib3", "httpx", "peewee"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)


def _table(rows: list[dict], cols: list[str]) -> str:
    w = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols} if rows else {c: len(c) for c in cols}
    line = lambda r: "  ".join(f"{r.get(c, '')!s:>{w[c]}}" for c in cols)
    return "\n".join([line({c: c for c in cols}), *(line(r) for r in rows)])


METRIC_COLS = ["strategy", "total_return_pct", "cagr_pct", "max_drawdown_pct", "sharpe", "sortino", "trades",
               "win_rate_pct", "profit_factor", "avg_trade", "charges_paid", "exposure_pct"]


# ---- commands ---------------------------------------------------------------
def cmd_check(a, cfg):
    import importlib
    import platform
    ok = True
    print(f"Python {platform.python_version()} ({'ok' if sys.version_info >= (3, 11) else 'NEED 3.11+'})")
    for pkg, required in [("pandas", 1), ("numpy", 1), ("sklearn", 1), ("yfinance", 1), ("requests", 1),
                          ("xgboost", 0), ("langgraph", 0), ("streamlit", 0), ("plotly", 0), ("backtrader", 0)]:
        try:
            m = importlib.import_module(pkg)
            print(f"  ok   {pkg} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = ok and not required
            print(f"  {'FAIL' if required else 'opt '} {pkg} not installed{'' if required else ' (optional)'}")
    from arena.safety import RealMoneyBlocked, SELF_TEST_URLS, check_url
    try:
        check_url("POST", SELF_TEST_URLS[0]); print("  FAIL order URL not blocked"); ok = False
    except RealMoneyBlocked:
        print("  ok   real-order URLs are blocked (paper trading only)")
    s = Store(cfg["data"]["db"])
    print(f"  ok   SQLite store at {s.path}")
    from .data.sources import DataSourceError, YahooSource
    try:
        df = YahooSource(retries=2).fetch("^NSEI", "1d", start="2024-01-01")
        print(f"  ok   Yahoo Finance reachable: ^NSEI last close {df['close'].iloc[-1]:.2f} on {df.index[-1].date()} (end-of-day)")
    except DataSourceError as e:
        ok = False
        print(f"  FAIL Yahoo Finance: {e}\n       Check your internet connection or proxy; see docs/QUANT_GUIDE.md 'Common errors'.")
    from .llm import Ollama
    print("  ok   Ollama model ready: " + cfg["llm"]["model"] if Ollama(cfg).available()
          else f"  opt  Ollama not running or model missing (optional): install from ollama.com, then `ollama pull {cfg['llm']['model']}`")
    print("\nAll required checks passed." if ok else "\nSome required checks failed.")
    return 0 if ok else 1


def cmd_fetch(a, cfg):
    s = Store(cfg["data"]["db"])
    res = pipeline.fetch(s, cfg, a.symbols.split(",") if a.symbols else None, a.interval)
    for r in res:
        print(f"  {r['symbol']:14s} {r['status']:5s} {r.get('rows', ''):>6} rows  last {r.get('last', '')}  {r.get('message', '')}"
              + (f"  [{'; '.join(r['issues'])}]" if r.get("issues") else ""))
    print(f"Source: Yahoo Finance via yfinance (unofficial, end-of-day for 1d). Stored in {s.path}")
    return 0 if all(r["status"] == "ok" for r in res) else 1


def cmd_import_csv(a, cfg):
    r = pipeline.import_csv(Store(cfg["data"]["db"]), a.file, a.symbol, a.interval)
    print(f"Imported {r['rows']} bars as {a.symbol}. Issues: {r['issues'] or 'none'}")


def cmd_quality(a, cfg):
    from .data import quality
    s = Store(cfg["data"]["db"])
    rows = []
    for sym in pipeline.all_symbols(cfg):
        r = quality.report(s.bars(sym, a.interval), sym, a.interval, stale_after_hours=cfg["data"]["stale_after_hours"])
        rows.append(r)
        print(f"{sym:14s} {r['status']:7s} rows={r['rows']:<5} {r.get('first', '')} -> {r.get('last', '')}  {r.get('kind', '')}")
        for w in r["warnings"]:
            print(f"    ! {w}")


def cmd_indicators(a, cfg):
    d = pipeline.prepared(Store(cfg["data"]["db"]), a.symbol)
    if d.empty:
        sys.exit(f"No data for {a.symbol}. Run `python -m quant fetch` first.")
    cols = ["close", "sma20", "ema21", "ema50", "ema200", "rsi14", "macd", "macd_signal", "macd_hist",
            "bb_up", "bb_lo", "atr14", "adx", "plus_di", "minus_di", "vwap", "vol_ratio"]
    print(d[cols].tail(a.rows).round(2).to_string())
    print(f"Source: {d.attrs.get('sources')} fetched {d.attrs.get('fetched_at')}. Prices adjusted for splits/dividends.")


def cmd_backtest(a, cfg):
    from . import backtest
    s = Store(cfg["data"]["db"])
    syms = a.symbols.split(",") if a.symbols else cfg["data"]["watchlist"]
    data = {x: pipeline.prepared(s, x) for x in syms}
    data = {k: v for k, v in data.items() if not v.empty}
    p_up = {k: s.predictions(k, cfg["ml"]["model"]) for k in data} if a.strategy in ("ml", "hybrid") else None
    r = backtest.run(data, a.strategy, cfg, p_up=p_up, start=a.start, end=a.end)
    bh = backtest.buy_and_hold({k: data[k] for k in r["symbols"]}, cfg, start=r["metrics"]["start"], end=r["metrics"]["end"])
    print(_table([{"strategy": a.strategy, **r["metrics"]}, {"strategy": "buy_and_hold", **bh["metrics"]}], METRIC_COLS))
    s.save_backtest("PORTFOLIO" if len(r["symbols"]) > 1 else r["symbols"][0], a.strategy, r["metrics"]["start"],
                    r["metrics"]["end"], r["metrics"], r["equity"], r["trades"], {"symbols": r["symbols"]})


def cmd_train(a, cfg):
    s = Store(cfg["data"]["db"])
    models = a.models.split(",") if a.models else [cfg["ml"]["model"]]
    res = pipeline.train(s, cfg, a.symbols.split(",") if a.symbols else None, models, a.task)
    cols = (["symbol", "model", "n", "base_rate", "accuracy", "auc", "log_loss", "mean_train_auc", "overfit_gap", "folds", "oos_start"]
            if a.task == "classification" else ["symbol", "model", "n", "mae", "direction_acc", "ic", "folds", "oos_start"])
    print(_table([r for r in res if "error" not in r], cols))
    for r in res:
        if "error" in r:
            print(f"  {r['symbol']} {r.get('model', '')}: {r['error']}")
    print("\nAll metrics are out-of-sample (purged expanding walk-forward). AUC 0.50 = no skill; "
          "accuracy must beat the base rate to mean anything.")


def cmd_compare(a, cfg):
    s = Store(cfg["data"]["db"])
    res = pipeline.compare(s, cfg, model=a.model)
    print(f"Same window for every strategy: {res['start']} to {res['end']} (out-of-sample), symbols {res['symbols']}")
    print(f"Costs: {cfg['costs']['slippage_bps']} bps slippage + full delivery charges per fill. Risk rules identical.\n")
    rows = sorted(({"strategy": k, **v["metrics"]} for k, v in res["results"].items()), key=lambda r: -r["sharpe"])
    print(_table(rows, METRIC_COLS))
    print("\nPast simulated results do not show future profitability. Survivorship bias: today's large caps were chosen with hindsight.")


def cmd_paper(a, cfg):
    from . import agents
    from .paper import PaperAccount
    if a.action == "reset":
        PaperAccount(cfg).reset(a.capital)
        print(f"Paper account reset to Rs {a.capital or cfg['paper']['capital']:,.0f} virtual capital.")
        return
    if a.action == "status":
        acct = PaperAccount(cfg)
        sim = acct.sim
        print(f"Paper account ({acct.mode}) as of {acct.last_date}: equity Rs {sim.total_equity:,.2f}, cash Rs {sim.cash:,.2f}")
        for p in sim.positions.values():
            print(f"  {p.symbol:14s} {p.qty:>6} @ {p.entry_price:.2f} since {p.entry_date}  stop {p.stop:.2f}  target {p.target:.2f}  last {p.last_price:.2f}")
        for k, o in sim.pending.items():
            print(f"  pending {o['side']} {k} (signal {o['signal_date']}) fills at the next open")
        for e in acct.events(15)[::-1]:
            print("  " + e["text"])
        return
    store = Store(cfg["data"]["db"])
    if a.action == "run":
        acct = PaperAccount(cfg)
        st = agents.run(store, cfg, acct, fetch=not a.no_fetch, use_llm=not a.no_llm, run_backtest=a.backtest)
        print(f"Agent cycle complete (as of {st['as_of']}, engine: {acct.last_agent_run()['engine']})")
        for e in st["log"]:
            print(f"  [{e['agent']}] {e['summary']}")
        return
    if a.action == "replay":
        import pandas as pd
        if not a.start:
            sys.exit("paper replay needs --start YYYY-MM-DD")
        path = str(Path(cfg["paper"]["db"]).with_name("paper_replay.db"))
        acct = PaperAccount(cfg, path)
        acct.reset(a.capital, mode="replay")
        train_end = (pd.Timestamp(a.start) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        print(f"Training models on data up to {train_end} only, so the replay period is unseen...")
        pipeline.train(store, cfg, end=train_end, prefix="replay:", save_predictions=False)
        days = [t for t in store.bars(pipeline.MARKET, start=a.start, end=a.end).index]
        for t in days:
            d = t.strftime("%Y-%m-%d")
            st = agents.run(store, cfg, acct, fetch=False, use_llm=False, end=d, model_prefix="replay:")
            fills = [e["summary"] for e in st["log"] if e["agent"] == "Paper Trading" and "Processed" in e["summary"]]
            print(f"{d}  equity Rs {acct.sim.total_equity:,.2f}  " + (fills[0].split(": ", 1)[1] if fills else ""))
        from . import metrics
        m = metrics.compute(acct.equity_curve(), acct.trades(), acct.sim.capital, acct.sim.charges_paid, acct.sim.slippage_paid)
        print("\nREPLAY RESULT (historical paper forward-test, not live):")
        print(json.dumps(m, indent=1))
        print(f"Saved to {path}; the dashboard shows it under 'Paper replay'.")


def cmd_dashboard(a, cfg):
    app = Path(__file__).with_name("dashboard.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
                            "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="quant", description="Free AI + algorithmic paper-trading agent (paper only)")
    p.add_argument("--config")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    f = sub.add_parser("fetch"); f.add_argument("--symbols"); f.add_argument("--interval", default="1d")
    c = sub.add_parser("import-csv"); c.add_argument("file"); c.add_argument("symbol"); c.add_argument("--interval", default="1d")
    q = sub.add_parser("quality"); q.add_argument("--interval", default="1d")
    i = sub.add_parser("indicators"); i.add_argument("symbol"); i.add_argument("--rows", type=int, default=5)
    b = sub.add_parser("backtest"); b.add_argument("--strategy", default="algo",
                                                   choices=["algo", "ml", "hybrid", "ma_crossover", "rsi_reversion", "macd_trend", "trend_following"])
    b.add_argument("--symbols"); b.add_argument("--start"); b.add_argument("--end")
    t = sub.add_parser("train"); t.add_argument("--models"); t.add_argument("--symbols")
    t.add_argument("--task", default="classification", choices=["classification", "regression"])
    cp = sub.add_parser("compare"); cp.add_argument("--model")
    pp = sub.add_parser("paper"); pp.add_argument("action", choices=["run", "status", "reset", "replay"])
    pp.add_argument("--no-fetch", action="store_true"); pp.add_argument("--no-llm", action="store_true")
    pp.add_argument("--backtest", action="store_true"); pp.add_argument("--capital", type=float)
    pp.add_argument("--start"); pp.add_argument("--end")
    d = sub.add_parser("dashboard"); d.add_argument("--port", type=int, default=8501)
    a = p.parse_args(argv)
    cfg = config.load(a.config)
    install_guards(cfg)  # refuses to start if paper_only is off; blocks broker SDK imports
    _setup_logging()
    fn = {"check": cmd_check, "fetch": cmd_fetch, "import-csv": cmd_import_csv, "quality": cmd_quality,
          "indicators": cmd_indicators, "backtest": cmd_backtest, "train": cmd_train, "compare": cmd_compare,
          "paper": cmd_paper, "dashboard": cmd_dashboard}[a.cmd]
    sys.exit(fn(a, cfg) or 0)


if __name__ == "__main__":
    main()
