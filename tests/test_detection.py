import pytest

from adie.detection import completeness, isolation_forest
from adie.faults.inject import apply as inject_fault
from tests.conftest import rebuild_dbt


def test_clean_baseline_has_no_completeness_signals(clean_duckdb):
    assert completeness.detect(clean_duckdb) == []


def test_isolation_forest_flags_a_schema_change_day(clean_duckdb):
    inject_fault(clean_duckdb, "schema_change", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    signals = isolation_forest.detect(clean_duckdb)
    flagged_days = {s.raw["pickup_date"] for s in signals}
    assert "2023-01-09" in flagged_days


@pytest.mark.parametrize("detector", [
    pytest.param("isolation_forest", marks=pytest.mark.xfail(
        strict=True, reason="legacy per-batch IsolationForest (contamination=0.1) spends its flag budget on the "
                            "2023-01-01/02 holiday and misses a fully broken join on the realistic demo window")),
    "hybrid",
])
def test_isolation_forest_identifies_the_driving_metric(clean_duckdb, detector):
    from adie.detection import selector

    inject_fault(clean_duckdb, "join_error", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    signals = selector.detect_row_level(clean_duckdb, name=detector, explain=False)[0]
    hit = [s for s in signals if s.raw["pickup_date"] == "2023-01-09"]
    assert hit, "join_error day was not flagged at all"
    assert hit[0].metric == "unmatched_zone_count"


def test_completeness_flags_exactly_the_deleted_day(clean_duckdb):
    inject_fault(clean_duckdb, "missing_partition", day="2023-01-09")
    rebuild_dbt(clean_duckdb)

    signals = completeness.detect(clean_duckdb)
    assert len(signals) == 1
    assert signals[0].raw["pickup_date"] == "2023-01-09"
    assert signals[0].severity.value == "critical"
