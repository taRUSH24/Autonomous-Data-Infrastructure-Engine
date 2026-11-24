"""Shared state for the LangGraph investigation/repair graph."""
from __future__ import annotations

from typing import Any, Literal, TypedDict


class RepairGraphState(TypedDict, total=False):
    incident_id: str
    duckdb_path: str
    dbt_project_dir: str
    manifest_path: str

    signals: list[dict]
    hypotheses: list[dict]
    proposed_action: dict | None
    llm_used: bool

    decision: Literal["approved", "rejected", "pending"]
    verification: dict | None
    final_status: Literal["applied", "rejected_by_human", "failed_verification", "no_signal"]
    log: list[str]
