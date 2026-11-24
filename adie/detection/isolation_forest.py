"""
Statistical/ML anomaly detector for plane 1.

Reuses scikit-learn's IsolationForest wholesale -- this module is a thin
adapter that pulls fct_trips_daily out of DuckDB, scores each day, and turns
outliers into Signal objects. No detection algorithm is implemented here.
"""
from __future__ import annotations

import duckdb
import pandas as pd
from sklearn.ensemble import IsolationForest

from adie.contracts import CheckKind, Severity, Signal

FEATURE_COLUMNS = [
    "trip_count",
    "avg_fare_amount",
    "avg_trip_distance_miles",
    "avg_total_amount",
    "unmatched_zone_count",
]

ASSET_KEY = "fct_trips_daily"


def load_daily_metrics(duckdb_path: str) -> pd.DataFrame:
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        df = con.execute(
            f"select * from main.fct_trips_daily order by pickup_date"
        ).fetchdf()
    finally:
        con.close()
    return df


def score_daily_metrics(
    df: pd.DataFrame,
    contamination: float = 0.1,
    random_state: int = 7,
) -> pd.DataFrame:
    """Returns df with an added `anomaly_score` (lower = more anomalous) and
    `is_anomaly` column, using sklearn's IsolationForest unmodified."""
    if len(df) < 4:
        # IsolationForest needs enough rows to build a meaningful forest;
        # below that, nothing is flagged rather than guessing.
        out = df.copy()
        out["anomaly_score"] = 0.0
        out["is_anomaly"] = False
        return out

    X = df[FEATURE_COLUMNS].fillna(0.0)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)

    out = df.copy()
    out["anomaly_score"] = model.decision_function(X)
    out["is_anomaly"] = model.predict(X) == -1
    return out


def _most_anomalous_feature(df: pd.DataFrame, flagged_index) -> str:
    """
    IsolationForest's decision_function scores the whole feature vector, not
    any one column. To point RCA at a *mechanism* rather than just "this day
    is odd", find which single feature deviates furthest (in z-scored units)
    from the other days -- this is the simple per-metric decomposition PyRCA
    calls epsilon-diagnosis, applied to our own feature set.
    """
    stats = df[FEATURE_COLUMNS].agg(["mean", "std"])
    row = df.loc[flagged_index, FEATURE_COLUMNS]
    z = ((row - stats.loc["mean"]) / stats.loc["std"].replace(0, 1)).abs()
    # A NaN feature (e.g. fare_amount gone entirely null for the day) is
    # itself the anomaly signature -- treat it as maximally anomalous, not
    # as "no signal", or a nulled-out column silently loses to any feature
    # with real but unremarkable variance.
    z = z.fillna(z.max(skipna=True) + 1 if z.notna().any() else 99.0)
    return z.idxmax()


def detect(duckdb_path: str) -> list[Signal]:
    df = load_daily_metrics(duckdb_path)
    scored = score_daily_metrics(df)

    signals: list[Signal] = []
    for idx, row in scored[scored["is_anomaly"]].iterrows():
        contributing_metric = _most_anomalous_feature(scored, idx)
        signals.append(
            Signal(
                asset_key=ASSET_KEY,
                check_name="isolation_forest_daily_metrics",
                kind=CheckKind.ANOMALY,
                severity=Severity.WARN if row["anomaly_score"] > -0.1 else Severity.CRITICAL,
                metric=contributing_metric,
                observed=float(row["anomaly_score"]),
                expected=0.0,
                message=(
                    f"{row['pickup_date'].date()} flagged by IsolationForest "
                    f"(score={row['anomaly_score']:.3f}, driven by {contributing_metric}): "
                    f"trip_count={row['trip_count']:.0f}, "
                    f"avg_fare=${row['avg_fare_amount']:.2f}, "
                    f"unmatched_zones={row['unmatched_zone_count']:.0f}"
                ),
                raw={"pickup_date": str(row["pickup_date"].date()), **{
                    c: float(row[c]) for c in FEATURE_COLUMNS
                }},
            )
        )
    return signals
