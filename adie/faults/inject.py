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


def _subset(fraction: float, alias: str = "") -> str:
    """Deterministic row subset for partial-severity faults: a stable hash
    of each trip's identifying columns, so the same `fraction` always hits
    the same rows (reproducible experiments) without needing a row id.
    fraction >= 1.0 returns an always-true predicate (the original,
    full-day behaviour of every fault)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return "true"
    a = f"{alias}." if alias else ""
    cutoff = int(round(fraction * 10000))
    return (
        f"(hash({a}tpep_pickup_datetime, {a}tpep_dropoff_datetime, {a}pulocationid, "
        f"{a}dolocationid, {a}trip_distance, {a}total_amount) % 10000) < {cutoff}"
    )


def _fault_duplicate_records(day: str, factor: int = 3, fraction: float = 1.0) -> FaultSpec:
    return FaultSpec(
        name="duplicate_records",
        description=(
            f"Duplicates {fraction:.0%} of raw trip rows for {day} {factor}x, "
            "simulating a replay/at-least-once ingestion bug."
        ),
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=(
            "insert into raw.trips select t.* from raw.trips t, "
            f"generate_series(1, {factor - 1}) g "
            f"where t.tpep_pickup_datetime::date = date '{day}' and {_subset(fraction, 't')}"
        ),
    )


def _fault_partial_partition(day: str, fraction: float = 0.5) -> FaultSpec:
    return FaultSpec(
        name="partial_partition",
        description=(
            f"Deletes {fraction:.0%} of raw trip rows for {day}, simulating a truncated / "
            "late-arriving upstream partition (the day exists but is incomplete)."
        ),
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=f"delete from raw.trips where tpep_pickup_datetime::date = date '{day}' and {_subset(fraction)}",
    )


def _fault_unit_change(day: str, fraction: float = 1.0) -> FaultSpec:
    return FaultSpec(
        name="unit_change",
        description=(
            f"Rewrites trip_distance in kilometres instead of miles for {fraction:.0%} of {day}'s raw "
            "rows, simulating an upstream producer silently changing units."
        ),
        ground_truth_asset="trips",
        mechanism="upstream_source_quality",
        sql=(
            "update raw.trips set trip_distance = cast(cast(trip_distance as double) * 1.609344 as varchar) "
            f"where tpep_pickup_datetime::date = date '{day}' and {_subset(fraction)}"
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


def _fault_join_error(day: str, fraction: float = 1.0) -> FaultSpec:
    return FaultSpec(
        name="join_error",
        description=f"Corrupts pickup_location_id to an out-of-range zone id for {fraction:.0%} of {day}'s rows, so trips_enriched's join silently drops the zone match.",
        ground_truth_asset="trips_enriched",
        mechanism="join_defect",
        sql=f"update raw.trips set pulocationid = '9999' where tpep_pickup_datetime::date = date '{day}' and {_subset(fraction)}",
    )


REGISTRY = {
    "missing_partition": _fault_missing_partition,
    "duplicate_records": _fault_duplicate_records,
    "schema_change": _fault_schema_change,
    "join_error": _fault_join_error,
    # Added for the hybrid-ensemble experiments (more fault *types* make
    # leave-one-fault-type-out evaluation meaningful). Both are upstream
    # data faults the existing reingest_partition repair can fix.
    "partial_partition": _fault_partial_partition,
    "unit_change": _fault_unit_change,
}

# Which fct_trips_daily feature each fault type is *designed* to move --
# the ground truth used to score explanation localization (does SHAP /
# LIME / ECOD point at the feature the fault actually corrupted?). Not used
# by any detector.
EXPECTED_FEATURE = {
    "duplicate_records": "trip_count",
    "partial_partition": "trip_count",
    "schema_change": "avg_fare_amount",
    "join_error": "unmatched_zone_count",
    "unit_change": "avg_trip_distance_miles",
}


def apply(duckdb_path: str, fault_name: str, day: str = "2023-01-03", **kwargs) -> FaultSpec:
    if fault_name not in REGISTRY:
        raise ValueError(f"unknown fault '{fault_name}', choose one of {list(REGISTRY)}")
    _validate_day(day)
    spec = REGISTRY[fault_name](day, **kwargs)

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
