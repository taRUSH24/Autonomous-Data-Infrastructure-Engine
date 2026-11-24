"""
Plane 0 (Substrate): the one asset that is NOT owned by dbt -- loading the
NYC TLC open-data feed into DuckDB's raw schema. Everything downstream of
this is dbt's `source('raw', 'trips')` / the `taxi_zone_lookup` seed.
"""

import dagster as dg
from dagster import AssetExecutionContext
import duckdb

from dagster_project.resources import DUCKDB_PATH, RAW_JSON_PATH


@dg.asset(
    group_name="substrate",
    description=(
        "Raw yellow-taxi trip records loaded verbatim from the NYC Open Data "
        "Socrata API (dataset 4b4i-vvec, '2023 Yellow Taxi Trip Data') into "
        "DuckDB's raw schema. This is the lineage boundary dbt's `source()` "
        "points at."
    ),
)
def raw_trips(context: AssetExecutionContext) -> dg.MaterializeResult:
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        con.execute("create schema if not exists raw")
        con.execute(
            "create or replace table raw.trips as "
            f"select * from read_json_auto('{RAW_JSON_PATH.as_posix()}', union_by_name=true)"
        )
        row_count = con.execute("select count(*) from raw.trips").fetchone()[0]
    finally:
        con.close()

    context.log.info(f"loaded {row_count} raw trip rows into {DUCKDB_PATH}")
    return dg.MaterializeResult(metadata={"row_count": row_count})
