from adie.contracts import Action, ActionKind, CheckKind, Hypothesis, Severity, Signal, Verification


def test_signal_round_trips_to_dict():
    sig = Signal(asset_key="fct_trips_daily", check_name="x", kind=CheckKind.ANOMALY,
                 severity=Severity.WARN, metric="trip_count", observed=1.0, expected=0.0)
    d = sig.to_dict()
    assert d["kind"] == "anomaly"
    assert d["severity"] == "warn"
    assert d["asset_key"] == "fct_trips_daily"


def test_hypothesis_default_fields_are_independent_lists():
    h1 = Hypothesis(asset_key="a", mechanism="m", score=1.0)
    h2 = Hypothesis(asset_key="b", mechanism="m", score=1.0)
    h1.evidence.append("x")
    assert h2.evidence == []  # mutable default must not be shared


def test_action_to_dict_serializes_enum():
    a = Action(kind=ActionKind.SQL_PATCH, target_asset="trips", description="d", payload={"sql_statements": ["select 1"]})
    assert a.to_dict()["kind"] == "sql_patch"


def test_verification_to_dict_nests_action():
    a = Action(kind=ActionKind.NO_OP, target_asset="x", description="d")
    v = Verification(action=a, passed=False)
    d = v.to_dict()
    assert d["action"]["kind"] == "no_op"
    assert d["passed"] is False
