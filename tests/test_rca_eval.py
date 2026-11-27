"""
The RCA eval harness: for each injected fault with a known ground-truth
asset, assert the ranker actually finds it -- this is what makes plane 3
falsifiable instead of merely plausible-looking (see docs/architecture).

schema_change is a deliberately honest exception: a value-level anomaly on
fct_trips_daily is genuinely ambiguous between "the source already had bad
values" and "the cast step corrupted them" from the aggregate alone, so the
scorer credits both candidate mechanisms rather than forcing a false-precise
top-1 answer. That fault is asserted at top-2, not top-1, and the test says
why rather than silently loosening the bar for everything.
"""
from __future__ import annotations

import pytest

from adie.detection import completeness, isolation_forest
from adie.faults.inject import apply as inject_fault
from adie.lineage.graph import LineageGraph
from adie.rca.scorer import rank_hypotheses
from tests.conftest import rebuild_dbt

FAULT_DAY = "2023-01-09"

TOP1_CASES = [
    ("missing_partition", "trips"),
    ("duplicate_records", "trips"),
    ("join_error", "trips_enriched"),
]


# Legacy per-batch IsolationForest on the REALISTIC demo window (uniform
# full-day sample with genuine weekday/holiday variation, which replaced the
# old constant-4,000-trips/day early-morning sample): contamination=0.1
# forces ~10% of every 14-day batch to be flagged, and the 2023-01-01/02 New
# Year holiday fills that budget with CRITICAL false alarms whose metric
# votes outweigh the real fault in RCA. Kept (strict) to document the
# measured weakness rather than deleted; see RESEARCH_REPORT.md section H4.
LEGACY_IF_HOLIDAY_FP = pytest.mark.xfail(
    strict=True, reason="legacy per-batch IsolationForest: New-Year holiday false alarms outvote the real fault")


def _detect(detector: str, db: str):
    from adie.detection import selector

    return selector.detect_all(db, name=detector, explain=False)[0]


@pytest.mark.parametrize("detector", [pytest.param("isolation_forest", marks=LEGACY_IF_HOLIDAY_FP), "hybrid"])
@pytest.mark.parametrize("fault_name,ground_truth_asset", TOP1_CASES[1:])
def test_rca_ranks_ground_truth_asset_first(clean_duckdb, manifest_path, fault_name, ground_truth_asset, detector):
    _rca_top1(clean_duckdb, manifest_path, fault_name, ground_truth_asset, detector)


@pytest.mark.parametrize("detector", ["isolation_forest", "hybrid"])
def test_rca_ranks_missing_partition_source_first(clean_duckdb, manifest_path, detector):
    _rca_top1(clean_duckdb, manifest_path, *TOP1_CASES[0], detector)


def _rca_top1(clean_duckdb, manifest_path, fault_name, ground_truth_asset, detector):
    inject_fault(clean_duckdb, fault_name, day=FAULT_DAY)
    rebuild_dbt(clean_duckdb)

    signals = _detect(detector, clean_duckdb)
    assert signals, f"{fault_name} produced no detectable signal"

    lg = LineageGraph.from_manifest(manifest_path)
    hyps = rank_hypotheses(lg, signals)
    assert hyps, "no hypotheses ranked"
    assert hyps[0].asset_key == ground_truth_asset, (
        f"{fault_name}: expected top-1 '{ground_truth_asset}', got '{hyps[0].asset_key}' "
        f"(top-3: {[(h.asset_key, h.score) for h in hyps[:3]]})"
    )


def test_rca_ranks_schema_change_source_in_top_2(clean_duckdb, manifest_path):
    """avg_fare_amount going anomalous is structurally ambiguous between the
    source (trips) and the transform step (stg_trips) -- both are credited,
    so we only require the true origin to surface in the top 2, not top 1."""
    inject_fault(clean_duckdb, "schema_change", day=FAULT_DAY)
    rebuild_dbt(clean_duckdb)

    signals = isolation_forest.detect(clean_duckdb) + completeness.detect(clean_duckdb)
    lg = LineageGraph.from_manifest(manifest_path)
    hyps = rank_hypotheses(lg, signals)

    top2 = [h.asset_key for h in hyps[:2]]
    assert "trips" in top2, f"expected 'trips' in top-2, got {[(h.asset_key, h.score) for h in hyps[:3]]}"


def test_rca_ranks_transformation_defect_correctly(manifest_path, tmp_path):
    """incorrect_transformation lives in code (stg_trips.sql), not in raw
    data -- apply it, rebuild, detect, verify stg_trips/transformation_defect
    ranks first, then restore the original model file no matter what."""
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
        signals = isolation_forest.detect(str(db_copy)) + completeness.detect(str(db_copy))
        lg = LineageGraph.from_manifest(manifest_path)
        hyps = rank_hypotheses(lg, signals)
        assert hyps[0].asset_key == "stg_trips"
        assert hyps[0].mechanism == "transformation_defect"
    finally:
        restore_transformation(str(DBT_PROJECT_DIR / "models"), original_sql)
