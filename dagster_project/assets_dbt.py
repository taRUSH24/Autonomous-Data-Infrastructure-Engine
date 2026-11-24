"""
Plane 0+2: wraps the dbt project as Dagster assets. dbt tests load as asset
checks automatically (enable_asset_checks defaults to True), which is what
makes the dbt DAG double as both the transformation layer and the
rule-based half of plane 1 (Detection) -- no separate rule engine needed.
"""

import dagster as dg
from dagster import AssetExecutionContext
from dagster_dbt import DagsterDbtTranslator, DagsterDbtTranslatorSettings, DbtCliResource, dbt_assets

from dagster_project.resources import dbt_project

dbt_translator = DagsterDbtTranslator(
    settings=DagsterDbtTranslatorSettings(
        enable_asset_checks=True,
        enable_source_tests_as_checks=True,
    )
)


@dbt_assets(
    manifest=dbt_project.manifest_path,
    project=dbt_project,
    dagster_dbt_translator=dbt_translator,
)
def adie_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()
