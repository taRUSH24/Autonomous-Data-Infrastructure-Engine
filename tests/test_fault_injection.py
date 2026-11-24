import duckdb
import pytest

from adie.faults.inject import REGISTRY, apply

FAULT_DAY = "2023-01-09"


@pytest.mark.parametrize("fault_name", list(REGISTRY))
def test_fault_applies_without_error(clean_duckdb, fault_name):
    spec = apply(clean_duckdb, fault_name, day=FAULT_DAY)
    assert spec.ground_truth_asset
    assert spec.mechanism


def test_missing_partition_deletes_the_day(clean_duckdb):
    apply(clean_duckdb, "missing_partition", day=FAULT_DAY)
    con = duckdb.connect(clean_duckdb, read_only=True)
    n = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}'"
    ).fetchone()[0]
    con.close()
    assert n == 0


def test_duplicate_records_multiplies_row_count(clean_duckdb):
    con = duckdb.connect(clean_duckdb, read_only=True)
    before = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}'"
    ).fetchone()[0]
    con.close()

    apply(clean_duckdb, "duplicate_records", day=FAULT_DAY)

    con = duckdb.connect(clean_duckdb, read_only=True)
    after = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}'"
    ).fetchone()[0]
    con.close()
    assert after == before * 3  # default factor


def test_schema_change_nulls_fare_amount(clean_duckdb):
    apply(clean_duckdb, "schema_change", day=FAULT_DAY)
    con = duckdb.connect(clean_duckdb, read_only=True)
    non_null = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}' and fare_amount is not null"
    ).fetchone()[0]
    con.close()
    assert non_null == 0


def test_join_error_corrupts_location_id(clean_duckdb):
    apply(clean_duckdb, "join_error", day=FAULT_DAY)
    con = duckdb.connect(clean_duckdb, read_only=True)
    corrupted = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}' and pulocationid = '9999'"
    ).fetchone()[0]
    total = con.execute(
        f"select count(*) from raw.trips where tpep_pickup_datetime::date = date '{FAULT_DAY}'"
    ).fetchone()[0]
    con.close()
    assert corrupted == total > 0


def test_incorrect_transformation_round_trips(tmp_path):
    from adie.faults.inject import apply_transformation_fault, restore_transformation
    from tests.conftest import DBT_PROJECT_DIR

    models_dir = str(DBT_PROJECT_DIR / "models")
    stg_trips = DBT_PROJECT_DIR / "models" / "staging" / "stg_trips.sql"
    original_on_disk = stg_trips.read_text()

    spec, returned_original = apply_transformation_fault(models_dir)
    try:
        assert "* 2" in stg_trips.read_text()
        assert returned_original == original_on_disk
    finally:
        restore_transformation(models_dir, returned_original)
    assert stg_trips.read_text() == original_on_disk
