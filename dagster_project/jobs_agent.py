"""
Exposes the LangGraph investigation/repair workflow (adie.agent.graph) as a
Dagster op + job, so an operator can trigger "investigate and propose a
repair" directly from the Dagster UI's Launchpad -- the asset graph and
check results are already visible there, so wiring the agent in as a job is
what Dagster's existing UI gives us for free, rather than building a
separate dashboard (requirement 9: only build custom UI if essential
capabilities are missing).

NOTE: approving the repair is an *out-of-band* step in this vertical slice
(the Launchpad does not render LangGraph's `interrupt()` payload for
in-place approval). The job runs the agent up to the interrupt, writes the
incident id into run metadata, and the operator resumes the same incident
from the CLI/notebook with adie.agent.graph + Command(resume=...). The
agent's approval gate is real (see adie/agent/graph.py); only the "approve
it by clicking in the Dagster UI" UX is out of scope for this slice.
"""

import os
import uuid

import dagster as dg
from dagster import OpExecutionContext
from langgraph.checkpoint.sqlite import SqliteSaver

from dagster_project.resources import DBT_PROJECT_DIR, DUCKDB_PATH, RAW_JSON_PATH


@dg.op(
    description="Runs the LangGraph investigation (gather evidence -> rank hypotheses -> propose repair) "
    "and pauses at the human-approval interrupt. See module docstring for how approval is resumed."
)
def investigate_and_propose(context: OpExecutionContext) -> str:
    from adie.agent.graph import make_graph

    incident_id = f"incident-{uuid.uuid4().hex[:8]}"
    checkpoint_path = str(DUCKDB_PATH.parent / "agent_checkpoints.sqlite")
    incident_db_path = str(DUCKDB_PATH.parent / "incidents.sqlite")

    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        graph = make_graph(
            duckdb_path=str(DUCKDB_PATH),
            dbt_project_dir=str(DBT_PROJECT_DIR),
            manifest_path=str(DBT_PROJECT_DIR / "target" / "manifest.json"),
            checkpointer=checkpointer,
            raw_json_path=str(RAW_JSON_PATH),
            incident_db_path=incident_db_path,
        )
        config = {"configurable": {"thread_id": incident_id}}
        result = graph.invoke(
            {
                "incident_id": incident_id,
                "duckdb_path": str(DUCKDB_PATH),
                "dbt_project_dir": str(DBT_PROJECT_DIR),
                "manifest_path": str(DBT_PROJECT_DIR / "target" / "manifest.json"),
                "log": [],
            },
            config=config,
        )

    interrupts = result.get("__interrupt__")
    if not interrupts:
        context.log.info(f"[{incident_id}] no anomaly found; nothing to propose.")
        return incident_id

    action = interrupts[0].value["action"]
    context.log.info(
        f"[{incident_id}] PAUSED FOR HUMAN APPROVAL\n"
        f"  proposed action: {action['description']}\n"
        f"  rationale: {action['payload'].get('rationale')}\n"
        f"  llm_used: {result.get('llm_used')}\n"
        f"  to resume: adie.agent.graph.make_graph(...).invoke(Command(resume=True/False), "
        f"config={{'configurable': {{'thread_id': '{incident_id}'}}}})"
    )
    return incident_id


@dg.job(description="Investigate the current state of fct_trips_daily and propose a repair (pauses for approval).")
def investigate_and_repair_job():
    investigate_and_propose()
