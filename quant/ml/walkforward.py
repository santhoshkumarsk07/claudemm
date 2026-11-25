"""Purged, expanding-window walk-forward validation.

    |---------------- train ----------------|purge|== test ==|
    |-------------------- train ---------------------|purge|== test ==|

- Train rows end `horizon + 1` rows before the test block, so no training label
  overlaps the period being predicted (labels look h+1 bars ahead).
- Each test block is predicted by a model that has never seen it, then the window
  grows. Concatenated test predictions form a genuine out-of-sample history.
- Never a random split: shuffling time series leaks the future into training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, roc_auc_score

from . import models


@dataclass
class WalkForwardResult:
    model: str
    task: str
    pred: pd.Series                    # out-of-sample P(up) (classification) or predicted return
    folds: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _clf_metrics(y, p) -> dict:
    out = {"n": int(len(y)), "base_rate": round(float(y.mean()), 3),
           "accuracy": round(accuracy_score(y, p >= 0.5), 3)}
    if len(set(y)) == 2:
        out["auc"] = round(roc_auc_score(y, p), 3)
        out["log_loss"] = round(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), 4)
    return out


def _reg_metrics(y, p) -> dict:
    ic = pd.Series(p).corr(pd.Series(y), method="spearman")
    return {"n": int(len(y)), "mae": round(mean_absolute_error(y, p), 5),
            "direction_acc": round(float(np.mean(np.sign(p) == np.sign(y))), 3), "ic": round(float(ic), 3)}


def run(X: pd.DataFrame, y: pd.Series, model_name: str, horizon: int, min_train: int = 750,
        test_bars: int = 63, task: str = "classification") -> WalkForwardResult:
    rows = X.dropna().index
    X, y = X.loc[rows], y.reindex(rows)
    n = len(X)
    if n < min_train + test_bars:
        raise ValueError(f"Only {n} usable rows; walk-forward needs at least {min_train + test_bars}.")
    preds, folds = [], []
    for t0 in range(min_train, n, test_bars):
        t1 = min(t0 + test_bars, n)
        train_end = t0 - horizon - 1            # purge: these labels are known before X[t0]'s close
        tr = y.iloc[:train_end].dropna().index
        if len(tr) < 200 or y.loc[tr].nunique() < 2 and task == "classification":
            continue
        m = models.make(model_name, task)
        m.fit(X.loc[tr].values, y.loc[tr].values)
        Xt = X.iloc[t0:t1]
        p = m.predict_proba(Xt.values)[:, 1] if task == "classification" else m.predict(Xt.values)
        preds.append(pd.Series(p, index=Xt.index))
        yt = y.iloc[t0:t1]
        ok = yt.notna().values
        fold = {"train_start": str(X.index[0].date()), "train_end": str(X.index[train_end - 1].date()),
                "test_start": str(Xt.index[0].date()), "test_end": str(Xt.index[-1].date()), "train_rows": len(tr)}
        if ok.sum() >= 10:
            fit_in = m.predict_proba(X.loc[tr].values)[:, 1] if task == "classification" else m.predict(X.loc[tr].values)
            if task == "classification":
                fold.update({f"test_{k}": v for k, v in _clf_metrics(yt.values[ok], p[ok]).items()})
                fold.update({f"train_{k}": v for k, v in _clf_metrics(y.loc[tr].values, fit_in).items() if k in ("auc", "accuracy")})
            else:
                fold.update({f"test_{k}": v for k, v in _reg_metrics(yt.values[ok], p[ok]).items()})
        folds.append(fold)
    pred = pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)
    yy = y.reindex(pred.index)
    ok = yy.notna()
    summary = (_clf_metrics(yy[ok].values, pred[ok].values) if task == "classification" else _reg_metrics(yy[ok].values, pred[ok].values))
    summary.update({"folds": len(folds), "oos_start": str(pred.index[0].date()) if len(pred) else None,
                    "oos_end": str(pred.index[-1].date()) if len(pred) else None})
    if task == "classification" and folds:
        tr_auc = [f["train_auc"] for f in folds if "train_auc" in f]
        te_auc = [f["test_auc"] for f in folds if "test_auc" in f]
        if tr_auc and te_auc:
            summary["mean_train_auc"] = round(float(np.mean(tr_auc)), 3)
            summary["mean_test_auc"] = round(float(np.mean(te_auc)), 3)
            summary["overfit_gap"] = round(summary["mean_train_auc"] - summary["mean_test_auc"], 3)
    return WalkForwardResult(model_name, task, pred, folds, summary)


def fit_final(X: pd.DataFrame, y: pd.Series, model_name: str, task: str = "classification"):
    """Model for live/paper predictions, trained on every row whose label is already known."""
    rows = X.dropna().index.intersection(y.dropna().index)
    m = models.make(model_name, task)
    m.fit(X.loc[rows].values, y.loc[rows].values)
    return m, rows[-1]
