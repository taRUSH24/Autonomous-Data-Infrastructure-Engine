"""
Research-experiment harness: builds many independent, labeled "arrangements"
of the real taxi warehouse (each a throwaway DuckDB file: N clean real days,
with zero or more of the five fault types injected on specific days), runs
the EXISTING dbt project + detectors + RCA scorer + agent against each one,
and returns ground-truth-labeled rows for ML training/evaluation.

Nothing here reimplements detection, RCA, or repair -- it only orchestrates
calls into adie.detection / adie.rca / adie.agent / adie.faults, the same
modules the live Dagster pipeline uses, so the experiment measures the real
system, not a simulation of it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from adie.contracts import validate_day
from adie.detection import completeness, isolation_forest
from adie.faults.inject import FaultSpec, apply as inject_fault

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DBT_PROJECT_DIR = REPO_ROOT / "dbt_project"
DBT_PROFILES_DIR = Path.home() / ".dbt"
MANIFEST_PATH = DBT_PROJECT_DIR / "target" / "manifest.json"

FAULT_TYPES = ["missing_partition", "duplicate_records", "schema_change", "join_error"]


@dataclass
class Arrangement:
    """One labeled experiment episode: a fully-built DuckDB warehouse plus
    which days (if any) were faulted and how."""
    name: str
    db_path: str
    days: list[str]
    fault_by_day: dict[str, FaultSpec] = field(default_factory=dict)  # day -> spec, only faulted days
    build_seconds: float = 0.0
    fault_inject_seconds: float = 0.0


def _load_raw_trips(db_path: str, json_path: str, days: list[str] | None = None) -> None:
    con = duckdb.connect(db_path)
    try:
        con.execute("create schema if not exists raw")
        con.execute(
            "create or replace table raw.trips as "
            f"select * from read_json_auto('{Path(json_path).as_posix()}', union_by_name=true)"
        )
        if days is not None:
            days_list = ", ".join(f"date '{validate_day(d)}'" for d in days)
            con.execute(f"delete from raw.trips where tpep_pickup_datetime::date not in ({days_list})")
    finally:
        con.close()


def _run_dbt_build(db_path: str, select: str = "+fct_trips_daily") -> float:
    # "+fct_trips_daily" pulls in every ancestor (the taxi_zone_lookup seed,
    # stg_trips, stg_zones, trips_enriched) too -- required the first time a
    # brand-new experiment DuckDB file is built, since none of those exist
    # yet the way they do in the long-lived demo warehouse.
    t0 = time.time()
    env = {**os.environ, "ADIE_DUCKDB_PATH": db_path}
    result = subprocess.run(
        ["dbt", "build", "--project-dir", str(DBT_PROJECT_DIR),
         "--profiles-dir", str(DBT_PROFILES_DIR), "--select", *select.split()],
        env=env, capture_output=True, text=True, timeout=120,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"dbt build failed:\n{result.stdout[-3000:]}\n{result.stderr[-1000:]}")
    return elapsed


def build_clean_arrangement(name: str, json_path: str, days: list[str], work_dir: str) -> Arrangement:
    """Loads the real, unmodified data for `days` and runs dbt once. No fault."""
    db_path = str(Path(work_dir) / f"{name}.duckdb")
    _load_raw_trips(db_path, json_path, days)
    build_seconds = _run_dbt_build(db_path)
    return Arrangement(name=name, db_path=db_path, days=days, build_seconds=build_seconds)


def build_fault_arrangement(
    name: str,
    json_path: str,
    days: list[str],
    fault_day_assignments: dict[str, str | tuple[str, dict]],  # day -> fault_type, or (fault_type, kwargs)
    work_dir: str,
) -> Arrangement:
    """Loads real data for `days`, injects one fault per (day, fault_type) in
    fault_day_assignments -- each on a different day so ground truth per day
    stays unambiguous -- then rebuilds once. A value may be a
    (fault_type, kwargs) tuple to pass severity parameters (e.g. fraction)."""
    db_path = str(Path(work_dir) / f"{name}.duckdb")
    _load_raw_trips(db_path, json_path, days)

    t0 = time.time()
    fault_by_day: dict[str, FaultSpec] = {}
    for day, assignment in fault_day_assignments.items():
        fault_type, kwargs = (assignment, {}) if isinstance(assignment, str) else assignment
        spec = inject_fault(db_path, fault_type, day=day, **kwargs)
        fault_by_day[day] = spec
    inject_seconds = time.time() - t0

    build_seconds = _run_dbt_build(db_path)
    return Arrangement(
        name=name, db_path=db_path, days=days, fault_by_day=fault_by_day,
        build_seconds=build_seconds, fault_inject_seconds=inject_seconds,
    )


def extract_day_features(arrangement: Arrangement):
    """Returns the raw fct_trips_daily rows (pandas DataFrame) for an
    arrangement, plus a ground-truth label/mechanism/asset per day."""
    con = duckdb.connect(arrangement.db_path, read_only=True)
    try:
        df = con.execute("select * from main.fct_trips_daily order by pickup_date").fetchdf()
    finally:
        con.close()

    df["pickup_date_str"] = df["pickup_date"].astype(str).str.slice(0, 10)
    df["is_fault"] = df["pickup_date_str"].isin(arrangement.fault_by_day.keys()).astype(int)
    df["fault_type"] = df["pickup_date_str"].map(
        lambda d: arrangement.fault_by_day[d].name if d in arrangement.fault_by_day else "none"
    )
    df["ground_truth_asset"] = df["pickup_date_str"].map(
        lambda d: arrangement.fault_by_day[d].ground_truth_asset if d in arrangement.fault_by_day else None
    )
    df["ground_truth_mechanism"] = df["pickup_date_str"].map(
        lambda d: arrangement.fault_by_day[d].mechanism if d in arrangement.fault_by_day else None
    )
    df["arrangement"] = arrangement.name
    return df


def run_detectors_timed(db_path: str):
    """Runs the EXISTING production detectors and times them -- this is the
    real `detection compute latency` reported in the research deliverables,
    not a fabricated number."""
    t0 = time.time()
    if_signals = isolation_forest.detect(db_path)
    if_seconds = time.time() - t0

    t0 = time.time()
    completeness_signals = completeness.detect(db_path)
    completeness_seconds = time.time() - t0

    return {
        "isolation_forest_signals": if_signals,
        "isolation_forest_seconds": if_seconds,
        "completeness_signals": completeness_signals,
        "completeness_seconds": completeness_seconds,
    }


def make_workdir(prefix: str) -> str:
    d = tempfile.mkdtemp(prefix=f"adie_experiment_{prefix}_")
    return d


def cleanup_workdir(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
