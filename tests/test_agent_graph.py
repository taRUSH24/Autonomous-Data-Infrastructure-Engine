"""
Structural tests for the LangGraph agent: compiles, pauses at the human
approval interrupt, resumes correctly in both directions, and never touches
the production DB on rejection. These run with or without ANTHROPIC_API_KEY
-- the deterministic fallback keeps the graph's control flow testable
without live model access (see adie/agent/graph.py's `_llm_available`).
"""
from __future__ import annotations

import hashlib

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from adie.agent.graph import make_graph
from adie.faults.inject import apply as inject_fault
from tests.conftest import rebuild_dbt


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _run_to_interrupt(duckdb_path, manifest_path, raw_json_path, checkpoint_path, incident_db_path, thread_id):
    checkpointer_cm = SqliteSaver.from_conn_string(checkpoint_path)
    checkpointer = checkpointer_cm.__enter__()
    graph = make_graph(
        duckdb_path=duckdb_path, dbt_project_dir="dbt_project", manifest_path=manifest_path,
        checkpointer=checkpointer, raw_json_path=raw_json_path, incident_db_path=incident_db_path,
    )
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"incident_id": thread_id, "duckdb_path": duckdb_path, "dbt_project_dir": "dbt_project",
         "manifest_path": manifest_path, "log": []},
        config=config,
    )
    return graph, config, result, checkpointer_cm


def test_graph_pauses_for_approval_with_a_real_action(clean_duckdb, manifest_path, raw_json_path, tmp_path):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    graph, config, result, cm = _run_to_interrupt(
        clean_duckdb, manifest_path, raw_json_path,
        str(tmp_path / "ckpt.sqlite"), str(tmp_path / "incidents.sqlite"), "t-approval",
    )
    try:
        assert "__interrupt__" in result
        action = result["__interrupt__"][0].value["action"]
        assert action["kind"] == "sql_patch"
        assert "2023-01-09" in action["description"]
    finally:
        cm.__exit__(None, None, None)


def test_graph_rejection_never_touches_production(clean_duckdb, manifest_path, raw_json_path, tmp_path):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)
    before_hash = _sha256(clean_duckdb)

    graph, config, result, cm = _run_to_interrupt(
        clean_duckdb, manifest_path, raw_json_path,
        str(tmp_path / "ckpt.sqlite"), str(tmp_path / "incidents.sqlite"), "t-reject",
    )
    try:
        final = graph.invoke(Command(resume=False), config=config)
        assert final["final_status"] == "rejected_by_human"
        assert _sha256(clean_duckdb) == before_hash
    finally:
        cm.__exit__(None, None, None)


def test_graph_approval_applies_and_heals_production(clean_duckdb, manifest_path, raw_json_path, tmp_path):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    graph, config, result, cm = _run_to_interrupt(
        clean_duckdb, manifest_path, raw_json_path,
        str(tmp_path / "ckpt.sqlite"), str(tmp_path / "incidents.sqlite"), "t-approve",
    )
    try:
        final = graph.invoke(Command(resume=True), config=config)
        assert final["final_status"] == "applied"
        assert final["verification"]["passed"] is True
    finally:
        cm.__exit__(None, None, None)

    import duckdb
    con = duckdb.connect(clean_duckdb, read_only=True)
    n = con.execute("select count(*) from main.fct_trips_daily where pickup_date = '2023-01-09'").fetchone()[0]
    con.close()
    assert n == 1  # the day is back


def test_transformation_defect_escalates_to_no_op_and_rejects(manifest_path, raw_json_path, tmp_path):
    import shutil

    from adie.faults.inject import apply_transformation_fault, restore_transformation
    from tests.conftest import BASELINE_DB, DBT_PROJECT_DIR

    if not BASELINE_DB.exists():
        pytest.skip("data/warehouse/adie.duckdb not found")

    db_copy = tmp_path / "test.duckdb"
    shutil.copy(BASELINE_DB, db_copy)

    spec, original_sql = apply_transformation_fault(str(DBT_PROJECT_DIR / "models"))
    try:
        rebuild_dbt(str(db_copy), select="stg_trips trips_enriched fct_trips_daily")
        graph, config, result, cm = _run_to_interrupt(
            str(db_copy), manifest_path, raw_json_path,
            str(tmp_path / "ckpt.sqlite"), str(tmp_path / "incidents.sqlite"), "t-transform",
        )
        try:
            action = result["__interrupt__"][0].value["action"]
            assert action["kind"] == "no_op"
            final = graph.invoke(Command(resume=True), config=config)
            assert final["final_status"] == "rejected_by_human"  # no_op is never auto-applied
        finally:
            cm.__exit__(None, None, None)
    finally:
        restore_transformation(str(DBT_PROJECT_DIR / "models"), original_sql)
