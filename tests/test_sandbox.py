"""
Verifies the core safety property: sandbox verification never writes to the
path it was given as "production" in these tests -- it only ever writes to
a throwaway clone it creates itself.
"""
from __future__ import annotations

import hashlib

from adie.agent.sandbox import verify_action
from adie.agent.tools import reingest_partition
from adie.faults.inject import apply as inject_fault
from tests.conftest import rebuild_dbt


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_verify_action_never_mutates_the_given_db_path(clean_duckdb, raw_json_path):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    before_hash = _sha256(clean_duckdb)
    action = reingest_partition("2023-01-09", raw_json_path)

    verify_action(action, clean_duckdb, "dbt_project")

    after_hash = _sha256(clean_duckdb)
    assert before_hash == after_hash, "sandbox verification wrote to the production-path argument"


def test_verify_action_reports_pass_on_a_correct_repair(clean_duckdb, raw_json_path):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    action = reingest_partition("2023-01-09", raw_json_path)
    v = verify_action(action, clean_duckdb, "dbt_project")

    assert v.passed is True
    assert v.checks_before["critical_count"] >= 1
    assert v.checks_after["critical_count"] == 0


def test_verify_action_reports_fail_on_a_wrong_repair(clean_duckdb, raw_json_path):
    """Re-ingesting the wrong day must not report success for the real fault."""
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    wrong_day_action = reingest_partition("2023-01-02", raw_json_path)  # unaffected day
    v = verify_action(wrong_day_action, clean_duckdb, "dbt_project")

    assert v.checks_after["critical_count"] >= 1  # 01-09 is still missing
