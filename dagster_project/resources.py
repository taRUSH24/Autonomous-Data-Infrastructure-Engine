from __future__ import annotations

import os
from pathlib import Path

from dagster_dbt import DbtCliResource, DbtProject

REPO_ROOT = Path(__file__).resolve().parent.parent
DBT_PROJECT_DIR = REPO_ROOT / "dbt_project"
DUCKDB_PATH = Path(os.environ.get("ADIE_DUCKDB_PATH", REPO_ROOT / "data" / "warehouse" / "adie.duckdb"))
RAW_JSON_PATH = REPO_ROOT / "data" / "raw" / "yellow_tripdata_2023-01_sample.json"

os.environ.setdefault("ADIE_DUCKDB_PATH", str(DUCKDB_PATH))
os.environ.setdefault("DBT_PROFILES_DIR", str(Path.home() / ".dbt"))

dbt_project = DbtProject(
    project_dir=DBT_PROJECT_DIR,
    profiles_dir=Path.home() / ".dbt",
)
dbt_project.prepare_if_dev()

dbt_resource = DbtCliResource(project_dir=dbt_project)
