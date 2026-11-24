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


@pytest.mark.parametrize("fault_name,ground_truth_asset", TOP1_CASES)
def test_rca_ranks_ground_truth_asset_first(clean_duckdb, manifest_path, fault_name, ground_truth_asset):
    inject_fault(clean_duckdb, fault_name, day=FAULT_DAY)
    rebuild_dbt(clean_duckdb)

    signals = isolation_forest.detect(clean_duckdb) + completeness.detect(clean_duckdb)
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
