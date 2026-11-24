"""
Plane 1 (Detection), the ML half: wraps the IsolationForest and calendar-
completeness detectors (adie.detection.*) as Dagster asset checks on
fct_trips_daily, so they show up in the Dagster UI exactly like dbt's
rule-based tests do.
"""
from __future__ import annotations

import dagster as dg

from adie.detection import completeness, isolation_forest
from dagster_project.resources import DUCKDB_PATH


@dg.asset_check(
    asset=dg.AssetKey(["fct_trips_daily"]),
    description="sklearn IsolationForest over the daily metrics profile (trip_count, avg_fare, unmatched_zones, ...).",
)
def isolation_forest_anomaly_check() -> dg.AssetCheckResult:
    signals = isolation_forest.detect(str(DUCKDB_PATH))
    return dg.AssetCheckResult(
        passed=len(signals) == 0,
        metadata={
            "anomalous_days": [s.raw.get("pickup_date") for s in signals],
            "signals": [s.message for s in signals],
        },
    )


@dg.asset_check(
    asset=dg.AssetKey(["fct_trips_daily"]),
    description="Calendar-completeness rule check: every day in range must have exactly one row.",
)
def calendar_completeness_check() -> dg.AssetCheckResult:
    signals = completeness.detect(str(DUCKDB_PATH))
    return dg.AssetCheckResult(
        passed=len(signals) == 0,
        metadata={"missing_days": [s.raw.get("pickup_date") for s in signals]},
    )
