"""
Rule-based detector for plane 1: calendar completeness on fct_trips_daily.

This is the counterpart to isolation_forest.py's row-level ML check. A
missing-partition fault deletes every row for a day, which means the day
never appears in fct_trips_daily at all -- there is no row for a row-level
anomaly detector to score. Only a check that reasons about the *expected*
calendar (what dbt source freshness / Elementary volume monitors do in
practice) can catch an absence. Implemented directly rather than pulled from
a library because it is a few lines of date-arithmetic, not an algorithm.
"""
from __future__ import annotations

import duckdb

from adie.contracts import CheckKind, Severity, Signal

ASSET_KEY = "fct_trips_daily"


def detect(duckdb_path: str) -> list[Signal]:
    import pandas as pd

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        df = con.execute("select pickup_date from main.fct_trips_daily").fetchdf()
    finally:
        con.close()

    if df.empty:
        return []

    present = set(pd.to_datetime(df["pickup_date"]).dt.date)
    start, end = min(present), max(present)
    expected = {
        (pd.Timestamp(start) + pd.Timedelta(days=i)).date()
        for i in range((end - start).days + 1)
    }
    missing = sorted(expected - present)

    return [
        Signal(
            asset_key=ASSET_KEY,
            check_name="calendar_completeness",
            kind=CheckKind.RULE,
            severity=Severity.CRITICAL,
            metric="missing_partition",
            observed=0.0,
            expected=1.0,
            message=f"{day} is missing from fct_trips_daily entirely (expected one row per day in range).",
            raw={"pickup_date": str(day)},
        )
        for day in missing
    ]
