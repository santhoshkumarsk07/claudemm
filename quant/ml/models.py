"""Model zoo. Deliberately small and regularised: daily data gives only ~250 rows a year,
so deep or wide models overfit. Fixed seeds make every run reproducible.

PyTorch is not used by default: for a few thousand rows of tabular features, gradient-boosted
trees and logistic regression are the stronger, cheaper baselines. Add a neural model only if
it beats them in walk-forward tests.
"""
from __future__ import annotations

from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
NAMES = ["logistic", "random_forest", "gradient_boosting", "xgboost"]


def make(name: str, task: str = "classification"):
    clf = task == "classification"
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000) if clf else Ridge(alpha=10.0))
    if name == "random_forest":
        kw = dict(n_estimators=300, max_depth=5, min_samples_leaf=25, max_features=0.5, n_jobs=-1, random_state=SEED)
        return RandomForestClassifier(**kw) if clf else RandomForestRegressor(**kw)
    if name == "gradient_boosting":
        kw = dict(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
        return HistGradientBoostingClassifier(**kw) if clf else HistGradientBoostingRegressor(**kw)
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise RuntimeError("xgboost is not installed: pip install xgboost") from e
        kw = dict(n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=2.0, random_state=SEED, n_jobs=2)
        return xgb.XGBClassifier(eval_metric="logloss", **kw) if clf else xgb.XGBRegressor(**kw)
    raise ValueError(f"Unknown model {name}. Choose from {NAMES}.")


def importances(model, columns: list[str]) -> dict[str, float]:
    est = model[-1] if hasattr(model, "steps") else model
    vals = getattr(est, "feature_importances_", None)
    if vals is None and hasattr(est, "coef_"):
        vals = abs(est.coef_).ravel()
    if vals is None:
        return {}
    tot = float(sum(vals)) or 1.0
    return dict(sorted(((c, round(float(v) / tot, 4)) for c, v in zip(columns, vals)), key=lambda kv: -kv[1]))
