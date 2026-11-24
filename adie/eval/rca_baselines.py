"""
Phase 2 comparison baselines for root-cause analysis. The lineage-constrained
method under test is adie.rca.scorer.rank_hypotheses (unchanged, imported
directly) -- this module only adds the two baselines it's compared against:

  1. statistical_no_lineage: ranks the SAME candidate assets by the
     magnitude of each asset's own table-level anomaly (row-count /
     null-rate deviation from a known-clean baseline), with no use of the
     dependency graph at all -- distance-blind, mechanism-blind.
  2. MLNoLineageRCA: a scikit-learn RandomForestClassifier that predicts
     the root-cause asset directly from the alerting Signal's (metric,
     severity) -- pattern-matching from training data, but never looking
     at the lineage graph either.

Both use the same fixed candidate-asset list as the lineage method, so the
three methods are compared on equal footing.
"""
from __future__ import annotations

import duckdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from adie.contracts import Signal

CANDIDATE_ASSETS = ["trips", "stg_trips", "stg_zones", "trips_enriched", "fct_trips_daily"]

TABLE_BY_ASSET = {
    "trips": "raw.trips",
    "stg_trips": "main.stg_trips",
    "stg_zones": "main.stg_zones",
    "trips_enriched": "main.trips_enriched",
    "fct_trips_daily": "main.fct_trips_daily",
}


def _table_health(db_path: str) -> dict[str, dict[str, float]]:
    """Generic, model-agnostic per-table health stats -- used only by the
    no-lineage statistical baseline. Real queries against the real tables,
    not synthetic numbers."""
    con = duckdb.connect(db_path, read_only=True)
    health = {}
    try:
        for asset, table in TABLE_BY_ASSET.items():
            row_count = con.execute(f"select count(*) from {table}").fetchone()[0]
            health[asset] = {"row_count": row_count}
    finally:
        con.close()
    return health


def statistical_no_lineage(db_path: str, baseline_health: dict[str, dict[str, float]]) -> list[str]:
    """Ranks candidates by |relative row-count deviation from a clean
    baseline| -- no graph distance, no mechanism reasoning. Returns asset
    keys ranked most-anomalous first."""
    current = _table_health(db_path)
    scored = []
    for asset in CANDIDATE_ASSETS:
        base = baseline_health.get(asset, {}).get("row_count", 0) or 1
        now = current.get(asset, {}).get("row_count", 0)
        deviation = abs(now - base) / base
        scored.append((asset, deviation))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [asset for asset, _ in scored]


class MLNoLineageRCA:
    """Supervised asset classifier trained on (metric, severity) -> ground
    truth asset, with zero access to the dependency graph. Reuses
    RandomForestClassifier wholesale."""

    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=200, random_state=7, class_weight="balanced")
        self._metric_encoder = LabelEncoder()
        self._severity_encoder = LabelEncoder()
        self._asset_encoder = LabelEncoder()
        self._fitted = False

    def _features(self, signals: list[Signal]) -> list[list[int]]:
        metrics = self._metric_encoder.transform([s.metric for s in signals])
        severities = self._severity_encoder.transform([s.severity.value for s in signals])
        return list(zip(metrics, severities))

    def fit(self, signals: list[Signal], ground_truth_assets: list[str]) -> "MLNoLineageRCA":
        all_metrics = sorted({s.metric for s in signals})
        all_severities = sorted({s.severity.value for s in signals})
        self._metric_encoder.fit(all_metrics)
        self._severity_encoder.fit(all_severities)
        self._asset_encoder.fit(sorted(set(ground_truth_assets) | set(CANDIDATE_ASSETS)))

        X = self._features(signals)
        y = self._asset_encoder.transform(ground_truth_assets)
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_ranked(self, signal: Signal) -> list[str]:
        """Returns CANDIDATE_ASSETS ranked by predicted probability,
        falling back to a fixed order for metric/severity values never
        seen in training (so an untrained signal degrades gracefully
        instead of raising)."""
        if not self._fitted:
            return list(CANDIDATE_ASSETS)
        try:
            metric_code = self._metric_encoder.transform([signal.metric])[0]
            severity_code = self._severity_encoder.transform([signal.severity.value])[0]
        except ValueError:
            return list(CANDIDATE_ASSETS)

        proba = self._model.predict_proba([[metric_code, severity_code]])[0]
        classes = self._asset_encoder.inverse_transform(self._model.classes_)
        ranked = [asset for _, asset in sorted(zip(proba, classes), reverse=True)]
        for asset in CANDIDATE_ASSETS:
            if asset not in ranked:
                ranked.append(asset)
        return ranked
