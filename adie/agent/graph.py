"""
Plane 4 (Agent): LangGraph state machine for investigate -> propose ->
approve -> sandbox-verify -> apply. LangGraph supplies the harness
(StateGraph, durable interrupts, checkpointing) -- this module only wires
nodes and edges, it does not implement an agent loop of its own.

Node restart semantics matter here: `await_approval` is the node containing
`interrupt()`, and LangGraph re-runs a node from its first line on resume.
Nothing before `interrupt()` in that node has a side effect, and no node
before it writes anywhere -- only `apply_repair`, which sits after the
interrupt AND after sandbox verification, touches the production database.
"""
from __future__ import annotations

import os
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from adie.agent.sandbox import apply_action, verify_action
from adie.agent.state import RepairGraphState
from adie.agent.tools import REPAIR_CATALOG_DESCRIPTION, deduplicate_raw_trips, reingest_partition
from adie.contracts import Action, ActionKind, Severity
from adie.control.incident_store import IncidentStore
from adie.detection import selector
from adie.lineage.graph import LineageGraph
from adie.rca.scorer import MECHANISM_BY_METRIC, rank_hypotheses

PROPOSER_SYSTEM_PROMPT = f"""\
You are the repair-proposal node of a data-pipeline self-healing agent.
You will be given ranked root-cause hypotheses for an anomaly in the
fct_trips_daily asset, each with a mechanism label and supporting evidence.

{REPAIR_CATALOG_DESCRIPTION}

Pick exactly one primitive. Respond with the day (YYYY-MM-DD) the anomaly is
scoped to (read it out of the evidence/signal messages), which primitive
you chose, and a one-sentence rationale citing the specific evidence that
led you there. If the top hypothesis's mechanism is transformation_defect
or unknown, you MUST choose no_op -- those require a human to change code,
not a data patch. Likewise choose no_op if anomalies are flagged on (nearly)
every day of the window: a systemic shift cannot be fixed by patching one
partition and points to a transformation/code change.
"""

SYSTEMIC_FRACTION = 0.8  # anomalous on >= 80% of the window's days -> systemic, escalate


def _systemic_anomaly(duckdb_path: str, signals: list[dict]) -> tuple[bool, int, int]:
    """True when row-level anomaly signals cover most days in fct_trips_daily
    -- the signature of a code-level (transformation) defect, not a single bad
    partition. Repair-policy guard only; RCA ranking is unchanged."""
    import duckdb

    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        n_days = con.execute("select count(distinct pickup_date) from main.fct_trips_daily").fetchone()[0]
    finally:
        con.close()
    flagged = {s["raw"].get("pickup_date") for s in signals if s.get("kind") == "anomaly"}
    return (n_days >= 7 and len(flagged) >= SYSTEMIC_FRACTION * n_days), len(flagged), n_days


class ProposedRepair:
    """Lightweight structured-output target. Kept as a plain class (not
    pydantic) at the call site in build_proposer() to avoid a hard pydantic
    version coupling; langchain's with_structured_output accepts a Pydantic
    model, so the real schema is defined there."""


def _build_proposer_schema():
    from pydantic import BaseModel, Field

    class Repair(BaseModel):
        day: str = Field(description="YYYY-MM-DD the anomaly is scoped to")
        primitive: Literal["reingest_partition", "deduplicate_raw_trips", "no_op"]
        rationale: str

    return Repair


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def make_graph(
    duckdb_path: str,
    dbt_project_dir: str,
    manifest_path: str,
    checkpointer,
    raw_json_path: str,
    incident_db_path: str = "data/warehouse/incidents.sqlite",
):
    store = IncidentStore(incident_db_path)

    def gather_evidence(state: RepairGraphState) -> RepairGraphState:
        # Detection: original IsolationForest or the hybrid ensemble (with
        # SHAP/LIME explanations attached to each signal's raw payload),
        # chosen by ADIE_DETECTOR -- plus the completeness rule check.
        signals, detector_used = selector.detect_all(state["duckdb_path"])
        # RCA stays lineage-based: explanations only supply the metric hint.
        lg = LineageGraph.from_manifest(state["manifest_path"])
        hyps = rank_hypotheses(lg, signals)

        store.record(state["incident_id"], "evidence_gathered", {
            "detector": detector_used,
            "signal_count": len(signals),
            "top_hypotheses": [h.to_dict() for h in hyps[:3]],
            "explanations": {s.raw.get("pickup_date"): s.raw["explanation"] for s in signals if s.raw.get("explanation")},
        })

        return {
            "signals": [s.to_dict() for s in signals],
            "detector": detector_used,
            "hypotheses": [h.to_dict() for h in hyps],
            "log": state.get("log", []) + [
                f"[{detector_used}] gathered {len(signals)} signal(s), {len(hyps)} hypothesis candidate(s)"],
        }

    def propose_repair(state: RepairGraphState) -> RepairGraphState:
        hyps = state["hypotheses"]
        signals = state["signals"]
        if not hyps:
            action = Action(kind=ActionKind.NO_OP, target_asset="fct_trips_daily", description="No anomaly detected.")
            store.record(state["incident_id"], "repair_proposed", {"action": action.to_dict(), "llm_used": False})
            return {"proposed_action": action.to_dict(), "llm_used": False,
                    "log": state["log"] + ["no hypotheses; proposing no_op"]}

        top = hyps[0]
        llm_used = _llm_available()

        if llm_used:
            from langchain_anthropic import ChatAnthropic

            schema = _build_proposer_schema()
            llm = ChatAnthropic(model="claude-opus-5", temperature=0).with_structured_output(schema)
            evidence_text = "\n".join(
                f"- {h['asset_key']} (mechanism={h['mechanism']}, score={h['score']}): {h['evidence']}"
                for h in hyps[:3]
            )
            signal_text = "\n".join(f"- {s['message']}" for s in signals)
            result = llm.invoke(
                [
                    ("system", PROPOSER_SYSTEM_PROMPT),
                    ("human", f"Signals:\n{signal_text}\n\nRanked hypotheses:\n{evidence_text}"),
                ]
            )
            primitive, day, rationale = result.primitive, result.day, result.rationale
        else:
            # Deterministic fallback used ONLY when no ANTHROPIC_API_KEY is
            # configured, so the graph remains testable end-to-end without
            # live model access. Never reported to the user as an LLM
            # decision -- `llm_used` on the Action/state tracks this.
            #
            # Use implicating_days -- the specific day(s) whose metric
            # actually earned the top hypothesis its mechanism-match bonus
            # -- not supporting_signals (a check_name like
            # "isolation_forest_daily_metrics" is shared across every day's
            # signal, so filtering on it can't tell which day is the real
            # one apart from unrelated baseline noise on other days).
            implicating = set(top.get("implicating_days", []))
            relevant = [s for s in signals if s["raw"].get("pickup_date") in implicating] or signals
            # Tie-break (after severity) by how unambiguous the signal's
            # metric is: trip_count/unmatched_zone_count/missing_partition
            # implicate exactly one mechanism, while a value-level metric
            # like avg_fare_amount is deliberately credited to two (source
            # vs transform) -- so a day whose metric is specific outranks a
            # day whose metric is inherently ambiguous, even at equal
            # severity. Without this, baseline noise on an ambiguous metric
            # can out-rank the real, specific signal just by list order.
            relevant.sort(
                key=lambda s: (
                    0 if s["severity"] == "critical" else 1,
                    len(MECHANISM_BY_METRIC.get(s["metric"], [])),
                )
            )
            lead_signal = relevant[0]
            day = lead_signal["raw"].get("pickup_date", "unknown")
            metric = lead_signal["metric"]

            systemic, n_flagged, n_days = _systemic_anomaly(state["duckdb_path"], signals)
            if top["mechanism"] in ("transformation_defect", "unknown") or systemic:
                primitive = "no_op"
            elif metric == "trip_count" and lead_signal["raw"].get("direction") != "below":
                # inflated volume -> dedupe. A *deficit* (hybrid detector
                # reports direction="below", e.g. a truncated partition) is
                # not fixable by dedupe, so it falls through to reingest.
                primitive = "deduplicate_raw_trips"
            else:
                primitive = "reingest_partition"
            rationale = (
                f"[deterministic fallback, no ANTHROPIC_API_KEY set] top hypothesis={top['asset_key']} "
                f"mechanism={top['mechanism']}, led by signal '{lead_signal['check_name']}' (metric={metric}) on {day}"
                + (f"; anomalous on {n_flagged}/{n_days} days -> systemic, escalating" if systemic else "")
            )

        if primitive == "reingest_partition":
            action = reingest_partition(day, raw_json_path)
        elif primitive == "deduplicate_raw_trips":
            action = deduplicate_raw_trips()
        else:
            action = Action(kind=ActionKind.NO_OP, target_asset=top["asset_key"],
                             description=f"No safe automated repair for mechanism={top['mechanism']}; escalate to human.")
        action.payload["rationale"] = rationale

        store.record(state["incident_id"], "repair_proposed", {"action": action.to_dict(), "llm_used": llm_used})
        return {
            "proposed_action": action.to_dict(),
            "llm_used": llm_used,
            "log": state["log"] + [f"proposed {action.kind.value} on {action.target_asset} (llm_used={llm_used}): {rationale}"],
        }

    def await_approval(state: RepairGraphState) -> Command[Literal["sandbox_verify", "reject"]]:
        action = state["proposed_action"]
        decision = interrupt({
            "question": "Approve this repair action?",
            "action": action,
        })
        store.record(state["incident_id"], "approval_decision", {"approved": bool(decision)})
        return Command(goto="sandbox_verify" if decision else "reject")

    def sandbox_verify(state: RepairGraphState) -> RepairGraphState:
        action_dict = state["proposed_action"]
        if action_dict["kind"] == ActionKind.NO_OP.value:
            return {"verification": {"passed": False, "reason": "no_op has nothing to verify"},
                    "log": state["log"] + ["no_op action: skipping sandbox verification, escalating"]}

        action = Action(
            kind=ActionKind(action_dict["kind"]),
            target_asset=action_dict["target_asset"],
            description=action_dict["description"],
            payload=action_dict["payload"],
        )
        v = verify_action(action, state["duckdb_path"], state["dbt_project_dir"])
        store.record(state["incident_id"], "sandbox_verification", v.to_dict())
        return {
            "verification": v.to_dict(),
            "log": state["log"] + [f"sandbox verification passed={v.passed}"],
        }

    def apply_repair(state: RepairGraphState) -> RepairGraphState:
        verification = state["verification"]
        if not verification or not verification.get("passed"):
            store.record(state["incident_id"], "final_status", {"status": "failed_verification"})
            return {"final_status": "failed_verification", "log": state["log"] + ["sandbox verification failed; NOT applying to production"]}

        action_dict = state["proposed_action"]
        action = Action(
            kind=ActionKind(action_dict["kind"]),
            target_asset=action_dict["target_asset"],
            description=action_dict["description"],
            payload=action_dict["payload"],
        )
        apply_log = apply_action(action, state["duckdb_path"], state["dbt_project_dir"])
        store.record(state["incident_id"], "final_status", {"status": "applied", "apply_log": apply_log})
        return {"final_status": "applied", "log": state["log"] + apply_log}

    def reject(state: RepairGraphState) -> RepairGraphState:
        store.record(state["incident_id"], "final_status", {"status": "rejected_by_human_or_no_op"})
        return {"final_status": "rejected_by_human", "log": state["log"] + ["repair rejected or escalated to human; no changes made"]}

    def route_after_verify(state: RepairGraphState) -> Literal["apply_repair", "reject"]:
        v = state.get("verification") or {}
        return "apply_repair" if v.get("passed") else "reject"

    graph = StateGraph(RepairGraphState)
    graph.add_node("gather_evidence", gather_evidence)
    graph.add_node("propose_repair", propose_repair)
    graph.add_node("await_approval", await_approval)
    graph.add_node("sandbox_verify", sandbox_verify)
    graph.add_node("apply_repair", apply_repair)
    graph.add_node("reject", reject)

    graph.add_edge(START, "gather_evidence")
    graph.add_edge("gather_evidence", "propose_repair")
    graph.add_edge("propose_repair", "await_approval")
    graph.add_conditional_edges("sandbox_verify", route_after_verify, ["apply_repair", "reject"])
    graph.add_edge("apply_repair", END)
    graph.add_edge("reject", END)

    return graph.compile(checkpointer=checkpointer)
