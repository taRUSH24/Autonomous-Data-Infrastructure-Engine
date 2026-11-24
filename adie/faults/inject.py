"""
Controlled fault injection against the `raw.trips` table in DuckDB.

Every fault is a plain SQL mutation, logged with a ground-truth label so the
eval harness (tests/test_rca.py) can check whether plane 3 actually ranks the
correct asset first -- this is what makes the RCA plane falsifiable rather
than merely plausible-looking.
"""
from __future__ import annotations

from dataclasses import dataclass

import duckdb

from adie.contracts import validate_day as _validate_day


@dataclass
class FaultSpec:
    name: str
    description: str
    ground_truth_asset: str   # which dbt asset the fault actually originates in
    mechanism: str            # should match adie.rca.scorer's mechanism vocabulary
    sql: str                  # mutation to run against the open connection


def _fault_missing_partition(day: str) -> FaultSpec:
    return FaultSpec(
        name="missing_partition",
        description=f"Deletes every raw trip row for {day}, simulating a dropped upstream partition.",
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=f"delete from raw.trips where tpep_pickup_datetime::date = date '{day}'",
    )


def _fault_duplicate_records(day: str, factor: int = 3) -> FaultSpec:
    return FaultSpec(
        name="duplicate_records",
        description=f"Duplicates every raw trip row for {day} {factor}x, simulating a replay/at-least-once ingestion bug.",
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=(
            "insert into raw.trips select t.* from raw.trips t, "
            f"generate_series(1, {factor - 1}) g "
            f"where t.tpep_pickup_datetime::date = date '{day}'"
        ),
    )


def _fault_schema_change(day: str) -> FaultSpec:
    return FaultSpec(
        name="schema_change",
        description=f"Nulls out fare_amount for {day}, simulating an upstream column going silently absent/renamed.",
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=f"update raw.trips set fare_amount = null where tpep_pickup_datetime::date = date '{day}'",
    )


def _fault_incorrect_transformation() -> FaultSpec:
    """
    This fault lives in dbt SQL, not raw data -- see
    faults/stg_trips_incorrect_transformation.sql.jinja. `apply()` below
    raises for this one; it's applied by swapping the model file, which the
    agent's sandbox-verification step is specifically designed to catch.
    """
    return FaultSpec(
        name="incorrect_transformation",
        description="Replaces stg_trips.sql with a version that double-counts fare_amount (bug lives in the transformation, not the source).",
        ground_truth_asset="stg_trips",
        mechanism="transformation_defect",
        sql="",  # applied via apply_transformation_fault(), not SQL
    )


def _fault_join_error(day: str) -> FaultSpec:
    return FaultSpec(
        name="join_error",
        description=f"Corrupts pickup_location_id to an out-of-range zone id for {day}, so trips_enriched's join silently drops the zone match.",
        ground_truth_asset="trips_enriched",
        mechanism="join_defect",
        sql=f"update raw.trips set pulocationid = '9999' where tpep_pickup_datetime::date = date '{day}'",
    )


REGISTRY = {
    "missing_partition": _fault_missing_partition,
    "duplicate_records": _fault_duplicate_records,
    "schema_change": _fault_schema_change,
    "join_error": _fault_join_error,
}


def apply(duckdb_path: str, fault_name: str, day: str = "2023-01-03", **kwargs) -> FaultSpec:
    if fault_name not in REGISTRY:
        raise ValueError(f"unknown fault '{fault_name}', choose one of {list(REGISTRY)}")
    _validate_day(day)
    spec = REGISTRY[fault_name](day, **kwargs) if fault_name == "duplicate_records" else REGISTRY[fault_name](day)

    con = duckdb.connect(duckdb_path)
    try:
        con.execute(spec.sql)
    finally:
        con.close()
    return spec


STG_TRIPS_BUGGED_SQL = """\
{{ config(materialized='view') }}

-- INJECTED FAULT: incorrect_transformation. fare_amount is accidentally
-- doubled here, which the source and fct_trips_daily's revenue columns both
-- reflect, but nothing upstream of this model is actually wrong.

with source as (
    select * from {{ source('raw', 'trips') }}
)

select
    cast(vendorid as integer)                      as vendor_id,
    cast(tpep_pickup_datetime as timestamp)         as pickup_at,
    cast(tpep_dropoff_datetime as timestamp)        as dropoff_at,
    cast(passenger_count as double)                 as passenger_count,
    cast(trip_distance as double)                   as trip_distance_miles,
    cast(ratecodeid as integer)                     as rate_code_id,
    store_and_fwd_flag,
    cast(pulocationid as integer)                   as pickup_location_id,
    cast(dolocationid as integer)                   as dropoff_location_id,
    cast(payment_type as integer)                   as payment_type,
    cast(fare_amount as double) * 2                 as fare_amount,
    cast(extra as double)                           as extra_amount,
    cast(mta_tax as double)                         as mta_tax_amount,
    cast(tip_amount as double)                      as tip_amount,
    cast(tolls_amount as double)                    as tolls_amount,
    cast(improvement_surcharge as double)           as improvement_surcharge,
    cast(total_amount as double)                    as total_amount,
    cast(congestion_surcharge as double)            as congestion_surcharge,
    date_trunc('day', cast(tpep_pickup_datetime as timestamp)) as pickup_date
from source
where tpep_pickup_datetime is not null
"""


def apply_transformation_fault(dbt_models_dir: str) -> tuple[FaultSpec, str]:
    """Overwrites stg_trips.sql in place; returns (spec, original_sql) so the
    caller can restore it. This is the one fault that lives in code, not data."""
    from pathlib import Path

    path = Path(dbt_models_dir) / "staging" / "stg_trips.sql"
    original = path.read_text()
    path.write_text(STG_TRIPS_BUGGED_SQL)
    return _fault_incorrect_transformation(), original


def restore_transformation(dbt_models_dir: str, original_sql: str) -> None:
    from pathlib import Path

    path = Path(dbt_models_dir) / "staging" / "stg_trips.sql"
    path.write_text(original_sql)
