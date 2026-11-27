from __future__ import annotations

import dagster as dg

from dagster_project.assets_dbt import adie_dbt_assets
from dagster_project.assets_detection import anomaly_explanation_report
from dagster_project.assets_substrate import raw_trips
from dagster_project.checks_detection import (
    calendar_completeness_check,
    hybrid_ensemble_anomaly_check,
    isolation_forest_anomaly_check,
)
from dagster_project.jobs_agent import investigate_and_repair_job
from dagster_project.resources import dbt_resource

defs = dg.Definitions(
    assets=[raw_trips, adie_dbt_assets, anomaly_explanation_report],
    asset_checks=[isolation_forest_anomaly_check, hybrid_ensemble_anomaly_check, calendar_completeness_check],
    jobs=[investigate_and_repair_job],
    resources={"dbt": dbt_resource},
)
