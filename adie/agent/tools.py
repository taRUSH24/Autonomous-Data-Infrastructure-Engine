"""
A small, fixed catalog of repair primitives the agent is allowed to choose
from -- deliberately not "let the LLM write arbitrary SQL against
production". Each primitive is parameterized (by day / asset) but the shape
of the fix is fixed and reviewed code, mirroring how production remediation
runbooks constrain an on-call agent to known-safe actions rather than
freeform commands.
"""
from __future__ import annotations

from adie.contracts import Action, ActionKind


def reingest_partition(day: str, raw_json_path: str) -> Action:
    """Repairs missing_partition, schema_change, and join_error: all three
    corrupt or remove rows for one day in raw.trips, and in every case the
    original source file still has the correct data for that day -- so the
    fix is to re-run ingestion for just that partition."""
    return Action(
        kind=ActionKind.SQL_PATCH,
        target_asset="trips",
        description=f"Re-ingest raw trip rows for {day} from the authoritative source file.",
        payload={
            "sql_statements": [
                f"delete from raw.trips where tpep_pickup_datetime::date = date '{day}'",
                (
                    "insert into raw.trips select * from "
                    f"read_json_auto('{raw_json_path}', union_by_name=true) "
                    f"where tpep_pickup_datetime::date = date '{day}'"
                ),
            ]
        },
    )


def deduplicate_raw_trips() -> Action:
    """Repairs duplicate_records: collapses exact-duplicate rows in raw.trips."""
    return Action(
        kind=ActionKind.SQL_PATCH,
        target_asset="trips",
        description="Remove exact-duplicate rows from raw.trips.",
        payload={
            "sql_statements": [
                "create or replace table raw.trips as select distinct * from raw.trips"
            ]
        },
    )


REPAIR_CATALOG_DESCRIPTION = """\
Available repair primitives (you must choose exactly one, or no_op if none fit):

1. reingest_partition(day) -- re-runs ingestion for one day's raw trip rows
   from the authoritative source file. Correct for: a day missing entirely,
   a column gone null for one day, or corrupted location/value columns for
   one day -- anything where the symptom is scoped to a specific date and
   the upstream source (not a transformation) is implicated.

2. deduplicate_raw_trips() -- removes exact-duplicate rows from raw.trips.
   Correct for: an inflated row/trip count with otherwise-normal per-row
   values, suggesting re-ingested or replayed records.

3. no_op -- propose this if the evidence does not clearly point to a safe,
   known repair (e.g. the mechanism is "transformation_defect" or
   "unknown" -- those require a code change, not a data patch, and must go
   to a human instead of being auto-repaired).
"""
