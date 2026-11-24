"""
The four anomaly-detection models compared in Phase 1. Every model is an
established implementation (scikit-learn's IsolationForest/RandomForest,
XGBoost's XGBClassifier) or a textbook statistical method (z-score
thresholding) -- nothing here is a novel detection algorithm.

IsolationForest here is a *fit-once* variant for a fair train/test
comparison against the supervised models. The *live* Dagster pipeline uses
adie.detection.isolation_forest.detect(), which fits per-batch (the
standard transductive way IsolationForest is normally deployed for
one-off monitoring runs) -- that function is untouched; this module adds
an offline fit/predict split purely for this research harness.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "trip_count",
    "avg_fare_amount",
    "avg_trip_distance_miles",
    "avg_total_amount",
    "unmatched_zone_count",
]


def _X(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].fillna(0.0).to_numpy()


class FittedModel:
    """Thin wrapper so every model exposes the same predict(df) -> (pred, latency_seconds) interface."""

    def __init__(self, name: str, predict_fn):
        self.name = name
        self._predict_fn = predict_fn

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        t0 = time.time()
        pred = self._predict_fn(df)
        return pred, time.time() - t0


def fit_isolation_forest(train_df: pd.DataFrame, contamination: float = 0.1, random_state: int = 7) -> FittedModel:
    # Matches adie.detection.isolation_forest's production default (0.1),
    # not sklearn's "auto" -- "auto" picks its offset from a fixed internal
    # heuristic, not the training batch, which makes it genuinely unstable
    # at the sample sizes this project's day-grain arrangements produce
    # (empirically confirmed: with a handful of days, "auto" sometimes
    # flagged a quiet day instead of the one with a 1500-row anomaly).
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(_X(train_df))

    def predict(df: pd.DataFrame) -> np.ndarray:
        return (model.predict(_X(df)) == -1).astype(int)

    return FittedModel("isolation_forest", predict)


def fit_random_forest(train_df: pd.DataFrame, labels: pd.Series, random_state: int = 7) -> FittedModel:
    model = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced")
    model.fit(_X(train_df), labels.to_numpy())

    def predict(df: pd.DataFrame) -> np.ndarray:
        return model.predict(_X(df)).astype(int)

    return FittedModel("random_forest", predict)


def fit_xgboost(train_df: pd.DataFrame, labels: pd.Series, random_state: int = 7) -> FittedModel:
    pos = max(int(labels.sum()), 1)
    neg = max(int((labels == 0).sum()), 1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=neg / pos,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(_X(train_df), labels.to_numpy())

    def predict(df: pd.DataFrame) -> np.ndarray:
        return model.predict(_X(df)).astype(int)

    return FittedModel("xgboost", predict)


class StatisticalBaseline:
    """Classic z-score thresholding: fit mean/std per feature on train data,
    flag a row if any feature's |z-score| exceeds the threshold. This is
    the required 'statistical anomaly-detection baseline' -- deliberately
    the simplest method in the comparison, not a from-scratch novel model."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "StatisticalBaseline":
        X = _X(train_df)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = _X(df)
        z = np.abs((X - self.mean_) / self.std_)
        return (z.max(axis=1) > self.threshold).astype(int)


def fit_statistical_baseline(train_df: pd.DataFrame, threshold: float = 3.0) -> FittedModel:
    baseline = StatisticalBaseline(threshold=threshold).fit(train_df)
    return FittedModel("statistical_zscore", baseline.predict)
