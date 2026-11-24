from adie.control.incident_store import IncidentStore


def test_record_and_timeline_round_trip(tmp_path):
    store = IncidentStore(tmp_path / "incidents.sqlite")
    store.record("inc-1", "evidence_gathered", {"signal_count": 2})
    store.record("inc-1", "final_status", {"status": "applied"})

    timeline = store.timeline("inc-1")
    assert [e["event_type"] for e in timeline] == ["evidence_gathered", "final_status"]
    assert timeline[0]["payload"]["signal_count"] == 2


def test_incidents_are_isolated_by_id(tmp_path):
    store = IncidentStore(tmp_path / "incidents.sqlite")
    store.record("inc-1", "x", {})
    store.record("inc-2", "y", {})
    assert len(store.timeline("inc-1")) == 1
    assert store.all_incident_ids() == ["inc-1", "inc-2"]
