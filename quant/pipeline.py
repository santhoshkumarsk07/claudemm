"""Glue used by the CLI, the agents and the dashboard."""
from __future__ import annotations

import logging

import pandas as pd

from . import backtest, indicators
from .data import quality
from .data.sources import CSVSource, DataSourceError, YahooSource
from .data.store import Store
from .ml import features, walkforward

log = logging.getLogger("quant")
MARKET = "^NSEI"


def all_symbols(cfg: dict) -> list[str]:
    return list(dict.fromkeys(cfg["data"]["indices"] + cfg["data"]["watchlist"]))


def fetch(store: Store, cfg: dict, symbols: list[str] | None = None, interval: str = "1d") -> list[dict]:
    """Download (or top up) bars from Yahoo. Failures are logged and reported, never filled with fake data."""
    src, out = YahooSource(), []
    for s in symbols or all_symbols(cfg):
        try:
            df = src.fetch(s, interval, cfg["data"]["start"] if interval == "1d" else None)
            if interval == "1d":
                df = quality.drop_unfinished_session(df)
            df, issues = quality.clean(df)
            n = store.upsert_bars(df, s, interval, src.name)
            store.log_fetch(s, interval, src.name, "ok", n, "; ".join(issues))
            out.append({"symbol": s, "status": "ok", "rows": n, "last": str(df.index[-1])[:19], "issues": issues})
        except DataSourceError as e:
            store.log_fetch(s, interval, src.name, "error", 0, str(e))
            out.append({"symbol": s, "status": "error", "message": str(e)})
    return out


def import_csv(store: Store, path: str, symbol: str, interval: str = "1d") -> dict:
    df, issues = quality.clean(CSVSource().fetch(path, interval))
    n = store.upsert_bars(df, symbol, interval, f"csv:{path.split('/')[-1]}")
    store.log_fetch(symbol, interval, "csv", "ok", n, "; ".join(issues))
    return {"symbol": symbol, "rows": n, "issues": issues}


def prepared(store: Store, symbol: str, end: str | None = None) -> pd.DataFrame:
    """Clean, split/dividend-adjusted daily bars with every indicator. Empty frame if no data."""
    raw = store.bars(symbol, "1d", end=end)
    if raw.empty:
        return raw
    attrs = dict(raw.attrs)
    raw = quality.drop_unfinished_session(raw)
    df, _ = quality.clean(raw)
    d = indicators.add_all(quality.adjusted(df), daily=True)
    d.attrs.update(attrs)
    return d


def dataset(store: Store, symbol: str, cfg: dict, end: str | None = None):
    d = prepared(store, symbol, end)
    mkt = prepared(store, MARKET, end) if symbol != MARKET else None
    X = features.build(d, mkt if mkt is not None and not mkt.empty else None)
    y, fwd = features.labels(d, cfg["ml"]["horizon"])
    return d, X, y, fwd


def train(store: Store, cfg: dict, symbols: list[str] | None = None, model_names: list[str] | None = None,
          task: str = "classification", end: str | None = None, prefix: str = "", save_predictions: bool = True) -> list[dict]:
    """Walk-forward every model on every symbol, save out-of-sample predictions,
    then fit the configured model on all known labels for live/paper use."""
    m, out = cfg["ml"], []
    model_names = model_names or [m["model"]]
    for s in symbols or cfg["data"]["watchlist"]:
        d, X, y, fwd = dataset(store, s, cfg, end)
        if d.empty:
            out.append({"symbol": s, "error": "no data; run fetch first"})
            continue
        target = y if task == "classification" else fwd
        for name in model_names:
            try:
                wf = walkforward.run(X, target, name, m["horizon"], m["min_train_bars"], m["test_bars"], task)
            except ValueError as e:
                out.append({"symbol": s, "model": name, "error": str(e)})
                continue
            if task == "classification" and save_predictions:
                store.save_predictions(s, name, wf.pred, "walk_forward")
            row = {"symbol": s, "model": name, "task": task, **wf.summary}
            if name == m["model"] and task == "classification":
                final, through = walkforward.fit_final(X, target, name, task)
                from .ml.models import importances
                imp = importances(final, list(X.columns))
                store.save_model(f"{prefix}{s}:{name}", final, str(through.date()),
                                 {"features": list(X.columns), "horizon": m["horizon"], "walk_forward": wf.summary,
                                  "importances": dict(list(imp.items())[:10])})
                row["final_trained_through"] = str(through.date())
            out.append(row)
            log.info("%s %s: %s", s, name, wf.summary)
    return out


def compare(store: Store, cfg: dict, symbols: list[str] | None = None, model: str | None = None,
            strategies_: tuple = ("algo", "ml", "hybrid", "ma_crossover", "rsi_reversion", "macd_trend", "trend_following"),
            save: bool = True) -> dict:
    """Run every strategy over the SAME out-of-sample window with the SAME costs and risk rules."""
    model = model or cfg["ml"]["model"]
    symbols = symbols or cfg["data"]["watchlist"]
    data = {s: prepared(store, s) for s in symbols}
    data = {s: d for s, d in data.items() if not d.empty}
    p_up = {s: store.predictions(s, model) for s in data}
    p_up = {s: p for s, p in p_up.items() if len(p)}
    if not p_up:
        raise RuntimeError("No walk-forward predictions found. Run `python -m quant train` first.")
    start = max(p.index[0] for p in p_up.values())   # common window where every symbol has ML predictions
    end = min(d.index[-1] for d in data.values())
    data = {s: data[s] for s in p_up}
    results = {"buy_and_hold": backtest.buy_and_hold(data, cfg, start=start, end=end)}
    for name in strategies_:
        results[name] = backtest.run(data, name, cfg, p_up=p_up, start=start, end=end)
    if save:
        for name, r in results.items():
            store.save_backtest("PORTFOLIO", name, start.date(), end.date(), r["metrics"], r["equity"], r["trades"],
                                {"symbols": list(data), "model": model, "costs": cfg["costs"], "risk": cfg["risk"]})
    return {"start": str(start.date()), "end": str(end.date()), "symbols": list(data), "model": model, "results": results}
