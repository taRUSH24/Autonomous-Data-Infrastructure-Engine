"""
Shared fixtures. Tests that need dbt (the manifest, a mutated warehouse)
mark themselves with @pytest.mark.dbt and get skipped automatically if dbt
hasn't been run yet in this checkout -- see `require_manifest` below.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT_DIR = REPO_ROOT / "dbt_project"
MANIFEST_PATH = DBT_PROJECT_DIR / "target" / "manifest.json"
BASELINE_DB = REPO_ROOT / "data" / "warehouse" / "adie.duckdb"
RAW_JSON = REPO_ROOT / "data" / "raw" / "yellow_tripdata_2023-01_sample.json"


@pytest.fixture(scope="session")
def manifest_path() -> str:
    if not MANIFEST_PATH.exists():
        pytest.skip("dbt_project/target/manifest.json not found -- run `dbt build` first")
    return str(MANIFEST_PATH)


@pytest.fixture(scope="session")
def raw_json_path() -> str:
    if not RAW_JSON.exists():
        pytest.skip("data/raw sample not found -- run scripts/fetch_data.py first")
    return str(RAW_JSON)


@pytest.fixture
def clean_duckdb(tmp_path) -> str:
    """A fresh copy of the known-good baseline warehouse, private to one test."""
    if not BASELINE_DB.exists():
        pytest.skip("data/warehouse/adie.duckdb not found -- run the ETL pipeline first")
    dest = tmp_path / "test.duckdb"
    shutil.copy(BASELINE_DB, dest)
    return str(dest)


def rebuild_dbt(duckdb_path: str, select: str = "trips_enriched fct_trips_daily") -> None:
    """Rebuilds the two models downstream of raw.trips after a fault/repair
    mutates raw data directly, so fct_trips_daily reflects the new state."""
    env = {**os.environ, "ADIE_DUCKDB_PATH": duckdb_path}
    result = subprocess.run(
        ["dbt", "build", "--project-dir", str(DBT_PROJECT_DIR),
         "--profiles-dir", str(Path.home() / ".dbt"), "--select", *select.split()],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"dbt build failed:\n{result.stdout[-3000:]}\n{result.stderr[-1000:]}"
