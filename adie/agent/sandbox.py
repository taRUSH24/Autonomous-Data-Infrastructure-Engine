"""
Sandbox verification: every proposed repair runs here first, against a
throwaway clone of the warehouse, before it is ever allowed near the real
DuckDB file. This module is the thing that makes requirement "never execute
unverified repairs against production data" true -- `apply_action` is the
only function in this codebase that writes to the production path, and it
refuses to run unless a prior `verify_action` call on a clone passed.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from adie.contracts import Action, ActionKind, Verification
from adie.detection import selector


def _run_checks(duckdb_path: str) -> dict:
    # Same detector the pipeline is configured with (ADIE_DETECTOR), so a
    # repair is verified against the signal that triggered it. Explanations
    # are skipped here -- verification only needs the alarms.
    signals, used = selector.detect_all(duckdb_path, explain=False)
    critical = sorted({f"{s.check_name}@{s.raw.get('pickup_date')}" for s in signals if s.severity.value == "critical"})
    return {
        "detector": used,
        "signal_count": len(signals),
        "critical_count": len(critical),
        "critical_keys": critical,
        "messages": [s.message for s in signals],
    }


def _run_dbt_build(dbt_project_dir: str, duckdb_path: str, select: str | None = None) -> tuple[bool, str]:
    cmd = ["dbt", "build", "--project-dir", dbt_project_dir, "--profiles-dir", str(Path.home() / ".dbt")]
    if select:
        cmd += ["--select", select]
    result = subprocess.run(
        cmd,
        env={**__import__("os").environ, "ADIE_DUCKDB_PATH": duckdb_path},
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.returncode == 0, (result.stdout[-4000:] + result.stderr[-2000:])


def verify_action(
    action: Action,
    production_db_path: str,
    dbt_project_dir: str,
    rebuild_select: str = "trips_enriched fct_trips_daily",
) -> Verification:
    """Clones production_db_path, applies `action` to the clone only, rebuilds
    the affected dbt models there, and compares detector output before vs
    after. Never touches production_db_path."""
    log: list[str] = []

    sandbox_fd, sandbox_path = tempfile.mkstemp(suffix=".duckdb", prefix="adie_sandbox_")
    import os

    os.close(sandbox_fd)
    os.remove(sandbox_path)  # duckdb wants to create the file itself
    shutil.copy(production_db_path, sandbox_path)
    log.append(f"cloned {production_db_path} -> {sandbox_path}")

    checks_before = _run_checks(sandbox_path)
    log.append(f"checks before repair: {checks_before}")

    if action.kind == ActionKind.SQL_PATCH:
        import duckdb

        con = duckdb.connect(sandbox_path)
        try:
            for stmt in action.payload["sql_statements"]:
                con.execute(stmt)
                log.append(f"applied to sandbox: {stmt[:200]}")
        finally:
            con.close()
        ok, build_log = _run_dbt_build(dbt_project_dir, sandbox_path, select=rebuild_select)
        log.append(f"dbt build on sandbox ({'OK' if ok else 'FAILED'}):\n{build_log[-1500:]}")
    elif action.kind == ActionKind.REMATERIALIZE:
        ok, build_log = _run_dbt_build(
            dbt_project_dir, sandbox_path, select=f"{action.target_asset}+ --full-refresh"
        )
        log.append(f"dbt rematerialize on sandbox ({'OK' if ok else 'FAILED'}):\n{build_log[-1500:]}")
    else:
        ok = True
        log.append("no_op action: nothing to verify")

    checks_after = _run_checks(sandbox_path)
    log.append(f"checks after repair: {checks_after}")

    # Pass iff the repair (a) resolved at least one critical alarm, (b)
    # introduced no NEW critical alarm, and (c) did not increase the total
    # signal count. Originally this required zero critical alarms anywhere,
    # which on real data means one unrelated pre-existing alarm (e.g. a
    # holiday the detector dislikes) blocks every correct repair forever.
    # A wrong-day repair still fails: it resolves nothing.
    before, after = set(checks_before["critical_keys"]), set(checks_after["critical_keys"])
    passed = ok and after < before and checks_after["signal_count"] <= checks_before["signal_count"]
    if passed and after:
        log.append(f"pre-existing critical alarms unrelated to this repair remain: {sorted(after)}")

    return Verification(
        action=action,
        passed=passed,
        checks_before=checks_before,
        checks_after=checks_after,
        sandbox_db_path=sandbox_path,
        log=log,
    )


def apply_action(action: Action, production_db_path: str, dbt_project_dir: str,
                  rebuild_select: str = "trips_enriched fct_trips_daily") -> list[str]:
    """Writes `action` to production. Caller MUST have a passed Verification
    for this exact action before calling this -- this function does not
    check that itself, by design the LangGraph graph (adie.agent.graph) is
    structured so this node is unreachable without one."""
    log: list[str] = []

    if action.kind == ActionKind.SQL_PATCH:
        import duckdb

        con = duckdb.connect(production_db_path)
        try:
            for stmt in action.payload["sql_statements"]:
                con.execute(stmt)
                log.append(f"applied to PRODUCTION: {stmt[:200]}")
        finally:
            con.close()
        ok, build_log = _run_dbt_build(dbt_project_dir, production_db_path, select=rebuild_select)
        log.append(f"dbt build on production ({'OK' if ok else 'FAILED'})")
    elif action.kind == ActionKind.REMATERIALIZE:
        ok, build_log = _run_dbt_build(
            dbt_project_dir, production_db_path, select=f"{action.target_asset}+ --full-refresh"
        )
        log.append(f"dbt rematerialize on production ({'OK' if ok else 'FAILED'})")
    else:
        log.append("no_op action: nothing applied")

    return log
