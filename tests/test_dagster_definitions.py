"""The Dagster code location must load: assets, the hybrid detection +
explanation asset, every asset check, and the agent job."""
from __future__ import annotations


def test_definitions_load_with_hybrid_detection(manifest_path):
    import dagster as dg

    from dagster_project.definitions import defs

    repo = defs.get_repository_def()
    keys = {k.to_user_string() for k in repo.asset_graph.get_all_asset_keys()}
    assert {"raw_trips", "fct_trips_daily", "anomaly_explanation_report"} <= keys
    check_names = {c.name for c in repo.asset_graph.asset_check_keys}
    assert {"isolation_forest_anomaly_check", "hybrid_ensemble_anomaly_check",
            "calendar_completeness_check"} <= check_names
    assert repo.get_job("investigate_and_repair_job")
    assert dg.AssetKey(["fct_trips_daily"]) in repo.asset_graph.get(
        dg.AssetKey(["anomaly_explanation_report"])).parent_keys
