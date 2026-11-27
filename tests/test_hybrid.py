"""
Tests for the hybrid ensemble, its calibration, explanations, the detector
switch, and the new fault types. Unit tests use a small synthetic feature
table (no dbt); integration tests use the real pre-trained artifact against
a real dbt-built warehouse and are skipped if either is missing.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from adie.detection import hybrid, selector
from adie.detection.hybrid import HybridEnsemble, TailCalibrator, ecod_raw, fit_ecod, to_matrix
from adie.detection.isolation_forest import FEATURE_COLUMNS
from adie.eval.hybrid_experiment import (
    ALPHA_UNSUP_GATE, MAX_FPR, choose_threshold, fit_ensemble, fpr_threshold, metrics,
)
from adie.faults.inject import apply as inject_fault
from tests.conftest import rebuild_dbt

FAULT_DAY = "2023-01-09"


# --------------------------------------------------------------------------- synthetic data

def _synthetic(n_clean: int, n_fault_per_type: int, seed: int, split: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "trip_count": rng.normal(1800, 150, n_clean),
        "avg_fare_amount": rng.normal(19.5, 0.8, n_clean),
        "avg_trip_distance_miles": rng.normal(3.4, 0.2, n_clean),
        "avg_total_amount": rng.normal(28.5, 1.0, n_clean),
        "unmatched_zone_count": np.zeros(n_clean),
    })
    rows = [base.assign(is_fault=0, fault_type="none")]
    pick = lambda: base.sample(n_fault_per_type, random_state=seed, replace=True).reset_index(drop=True)
    rows.append(pick().assign(trip_count=lambda d: d.trip_count * 2, is_fault=1, fault_type="duplicate_records"))
    rows.append(pick().assign(unmatched_zone_count=lambda d: d.trip_count * 0.5, is_fault=1, fault_type="join_error"))
    rows.append(pick().assign(avg_fare_amount=np.nan, is_fault=1, fault_type="schema_change"))
    rows.append(pick().assign(avg_trip_distance_miles=lambda d: d.avg_trip_distance_miles * 1.609,
                              is_fault=1, fault_type="unit_change"))
    df = pd.concat(rows, ignore_index=True)
    df["split"] = split
    df["pickup_date"] = pd.date_range("2023-01-01", periods=len(df)).astype(str)
    return df


@pytest.fixture(scope="module")
def fitted():
    train = _synthetic(150, 30, 1, "train")
    val = _synthetic(60, 12, 2, "val")
    types = ["duplicate_records", "join_error", "schema_change", "unit_change"]
    return fit_ensemble(train, val, types, ["isolation_forest", "ecod"]), train, val


# --------------------------------------------------------------------------- calibration

def test_tail_calibrator_is_monotone_bounded_and_unsaturated():
    cal = TailCalibrator().fit(np.arange(10, dtype=float))
    s = np.array([-5.0, 0.0, 4.5, 9.0, 9.5, 12.0, 20.0])
    u = cal.transform(s)
    assert np.all(np.diff(u) >= 0)
    assert u.min() >= 0 and u.max() < 1
    assert u[-1] > u[-2] > u[-3], "scores beyond the clean max must keep ranking (no saturation)"


def test_choose_threshold_respects_fpr_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(100), np.ones(20)].astype(int)
    s = np.r_[rng.normal(0, 1, 100), rng.normal(1.0, 1, 20)]
    t = choose_threshold(s, y, max_fpr=0.05)
    assert ((s >= t) & (y == 0)).sum() / 100 <= 0.05


def test_fpr_threshold_allows_at_most_alpha_clean_days():
    s = np.arange(100, dtype=float)
    t = fpr_threshold(s, 0.02)
    assert (s >= t).sum() == 2


def test_ecod_row_wise_scoring_has_no_batch_leakage():
    rng = np.random.default_rng(0)
    m = fit_ecod(rng.normal(size=(80, 5)))
    x = rng.normal(size=(1, 5))
    alone = ecod_raw(m, x)[0]
    with_outliers = ecod_raw(m, np.vstack([x, rng.normal(50, 1, size=(20, 5))]))[0]
    assert alone == pytest.approx(with_outliers)


# --------------------------------------------------------------------------- ensemble

def test_ensemble_validation_fpr_within_budget(fitted):
    m, _, val = fitted
    Xv, yv = to_matrix(val), val.is_fault.to_numpy()
    for name, s in m.candidate_scores(Xv).items():
        pred = s >= m.threshold(name)
        assert metrics(yv, pred)["fpr"] <= MAX_FPR + 1e-9, name
    u_clean = m.unsup_score(to_matrix(val[val.is_fault == 0]))
    assert (u_clean >= m.t_unsup).mean() <= ALPHA_UNSUP_GATE + 1e-9


def test_gated_fusion_fires_on_either_gate(fitted):
    m, _, _ = fitted
    p = np.array([m.t_sup + 1e-6, 0.0, 0.0])
    u = np.array([0.0, m.t_unsup + 1e-6, 0.0])
    assert list(m.fuse_gated(p, u) >= 1.0) == [True, True, False]


def test_unsupervised_branch_flags_a_fault_type_it_was_never_shown(fitted):
    """A never-seen pattern that corrupts several features at once (a
    'garbage day': volume, fares and distances all collapse) must be caught
    by the novelty gate with no supervised label for it. NOTE: a corruption
    confined to ONE feature is not guaranteed to pass the gate -- IF and
    ECOD scores saturate beyond the training range (see RESEARCH_REPORT)."""
    m, train, _ = fitted
    novel = train[train.is_fault == 0].head(5).assign(
        trip_count=5.0, avg_fare_amount=1.0, avg_trip_distance_miles=0.1, avg_total_amount=2.0)
    out = m.predict_frame(novel)
    assert out.is_anomaly.all()
    assert set(out.branch) <= {"unsupervised_novel", "both"}


def test_clean_days_mostly_pass(fitted):
    m, _, _ = fitted
    clean = _synthetic(200, 0, 99, "x")
    clean = clean[clean.is_fault == 0]
    assert m.predict_frame(clean).is_anomaly.mean() <= 0.1


def test_artifact_roundtrip(fitted, tmp_path):
    m, _, val = fitted
    path = m.save(tmp_path / "h.joblib")
    m2 = hybrid.load(path)
    np.testing.assert_allclose(m.predict_frame(val).ensemble_score, m2.predict_frame(val).ensemble_score)
    assert (tmp_path / "metadata.json").exists()


# --------------------------------------------------------------------------- explanations

def test_explanations_give_full_top5_and_point_at_the_corrupted_feature(fitted):
    from adie.explain.attributions import explain_rows

    m, train, _ = fitted
    row = train[train.fault_type == "join_error"].head(1)
    e = explain_rows(m, to_matrix(row), with_lime=True)[0]
    for key in ("supervised_shap_top5", "ecod_top5", "lime_top5"):
        assert len(e[key]) == 5 and {d["feature"] for d in e[key]} == set(FEATURE_COLUMNS)
    assert e["supervised_shap_top5"][0]["feature"] == "unmatched_zone_count"
    assert e["ecod_top5"][0]["feature"] == "unmatched_zone_count"
    assert "causal" in e["disclaimer"]


def test_ecod_contributions_sum_to_score(fitted):
    from adie.explain.attributions import ecod_contributions

    m, train, _ = fitted
    X = to_matrix(train.head(4))
    np.testing.assert_allclose(ecod_contributions(m.models["ecod"], X).sum(1), ecod_raw(m.models["ecod"], X))


# --------------------------------------------------------------------------- selector

def test_selector_falls_back_to_isolation_forest_when_artifact_missing(clean_duckdb, monkeypatch, tmp_path):
    monkeypatch.setenv("ADIE_DETECTOR", "hybrid")
    monkeypatch.setenv("ADIE_HYBRID_ARTIFACT", str(tmp_path / "missing.joblib"))
    _, used = selector.detect_row_level(clean_duckdb)
    assert used == "isolation_forest (fallback)"


def test_selector_rejects_unknown_detector(monkeypatch):
    monkeypatch.setenv("ADIE_DETECTOR", "magic")
    with pytest.raises(ValueError):
        selector.configured_detector()


# --------------------------------------------------------------------------- new fault types

def _day_stats(db):
    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(
            "select count(*), avg(cast(trip_distance as double)) from raw.trips "
            f"where tpep_pickup_datetime::date = date '{FAULT_DAY}'").fetchone()
    finally:
        con.close()


def test_partial_partition_removes_roughly_the_requested_fraction(clean_duckdb):
    n0, _ = _day_stats(clean_duckdb)
    inject_fault(clean_duckdb, "partial_partition", day=FAULT_DAY, fraction=0.5)
    n1, _ = _day_stats(clean_duckdb)
    assert 0.4 < 1 - n1 / n0 < 0.6


def test_unit_change_scales_distance(clean_duckdb):
    _, d0 = _day_stats(clean_duckdb)
    inject_fault(clean_duckdb, "unit_change", day=FAULT_DAY)
    _, d1 = _day_stats(clean_duckdb)
    assert d1 / d0 == pytest.approx(1.609344, rel=1e-6)


# --------------------------------------------------------------------------- integration (real artifact)

@pytest.fixture
def artifact():
    if not hybrid.DEFAULT_ARTIFACT_PATH.exists():
        pytest.skip("models/hybrid_ensemble artifact not trained -- run scripts/train_hybrid_ensemble.py")
    return hybrid.DEFAULT_ARTIFACT_PATH


@pytest.mark.parametrize("fault,kwargs,metric", [
    ("join_error", {}, "unmatched_zone_count"),
    ("duplicate_records", {}, "trip_count"),
    ("schema_change", {}, "avg_fare_amount"),
])
def test_hybrid_detects_and_localizes_real_faults(clean_duckdb, artifact, fault, kwargs, metric):
    inject_fault(clean_duckdb, fault, day=FAULT_DAY, **kwargs)
    rebuild_dbt(clean_duckdb)
    signals = hybrid.detect(clean_duckdb, artifact_path=artifact)
    hit = [s for s in signals if s.raw["pickup_date"] == FAULT_DAY]
    assert hit, f"{fault} not flagged"
    assert hit[0].metric == metric
    assert hit[0].raw["explanation"]["primary_top5"]


def test_hybrid_rca_handoff_finds_join_defect(clean_duckdb, artifact, manifest_path):
    from adie.lineage.graph import LineageGraph
    from adie.rca.scorer import rank_hypotheses

    inject_fault(clean_duckdb, "join_error", day=FAULT_DAY)
    rebuild_dbt(clean_duckdb)
    signals = [s for s in hybrid.detect(clean_duckdb, artifact_path=artifact) if s.raw["pickup_date"] == FAULT_DAY]
    hyps = rank_hypotheses(LineageGraph.from_manifest(manifest_path), signals)
    assert hyps[0].asset_key == "trips_enriched"


def test_agent_reingests_a_truncated_partition_instead_of_deduplicating(
        clean_duckdb, artifact, manifest_path, raw_json_path, tmp_path, monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver

    from adie.agent.graph import make_graph

    monkeypatch.setenv("ADIE_DETECTOR", "hybrid")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    inject_fault(clean_duckdb, "partial_partition", day=FAULT_DAY, fraction=0.75)
    rebuild_dbt(clean_duckdb)
    with SqliteSaver.from_conn_string(str(tmp_path / "cp.sqlite")) as cp:
        graph = make_graph(duckdb_path=clean_duckdb, dbt_project_dir="dbt_project", manifest_path=manifest_path,
                           checkpointer=cp, raw_json_path=raw_json_path, incident_db_path=str(tmp_path / "i.sqlite"))
        res = graph.invoke({"incident_id": "t", "duckdb_path": clean_duckdb, "dbt_project_dir": "dbt_project",
                            "manifest_path": manifest_path, "log": []},
                           config={"configurable": {"thread_id": "t"}})
    assert res["detector"] == "hybrid"
    action = res["__interrupt__"][0].value["action"]
    assert FAULT_DAY in action["description"] and "Re-ingest" in action["description"]
