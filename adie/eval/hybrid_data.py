"""
Labelled dataset for the hybrid-ensemble experiments, built by running the
REAL pipeline (raw load -> fault injection -> dbt build -> fct_trips_daily)
through adie.eval.harness -- the same path the earlier experiments used.

Fixes two evaluation issues found in the earlier harness:
  1. pseudo-replication: every fault arrangement re-contributed all of its
     *unfaulted* days as extra "clean" rows, so each real clean day appeared
     ~20x in train and test. Here clean rows come from the clean arrangement
     exactly once; fault arrangements contribute only their faulted days.
  2. trivially-separable faults: every fault was full-severity (3x rows,
     100% of zone ids broken) on data whose trip_count was constant by
     construction. Here faults have graded severities and trip_count varies
     with real demand.

Splits are chronological (see scripts/fetch_hybrid_data.py); no calendar
day ever appears in more than one split.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pandas as pd

from adie.detection.isolation_forest import FEATURE_COLUMNS
from adie.eval import harness
from adie.faults.inject import EXPECTED_FEATURE

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "experiments" / "data" / "hybrid"
FEATURES_CACHE = DATA_DIR / "features.parquet"

# Graded severities per row-level fault type (fraction of the day's rows hit).
SEVERITIES: dict[str, list[dict]] = {
    "duplicate_records": [{"factor": 2, "fraction": f} for f in (0.1, 0.25, 0.5, 1.0)],
    "partial_partition": [{"fraction": f} for f in (0.1, 0.25, 0.5, 0.75)],
    "join_error": [{"fraction": f} for f in (0.02, 0.1, 0.5, 1.0)],
    "unit_change": [{"fraction": f} for f in (0.25, 0.5, 1.0)],
    # A *partial* null of fare_amount is invisible in avg(fare_amount)
    # (SQL avg ignores nulls), so only the full-day variant is used.
    "schema_change": [{}],
}
ROW_FAULT_TYPES = list(SEVERITIES)

# Real days whose distribution legitimately differs from an ordinary day --
# none of these should trigger a repair.
LEGIT_EVENTS = {
    "2023-01-16": "MLK Day", "2023-02-20": "Presidents Day", "2023-05-29": "Memorial Day",
    "2023-06-19": "Juneteenth", "2023-07-04": "Independence Day", "2023-09-04": "Labor Day",
    "2023-09-29": "Flash-flood emergency", "2023-10-09": "Columbus Day", "2023-11-05": "NYC Marathon",
    "2023-11-23": "Thanksgiving", "2023-11-24": "Black Friday", "2023-12-24": "Christmas Eve",
    "2023-12-25": "Christmas Day", "2023-12-31": "New Year's Eve",
}

ARRANGEMENTS_PER_SPLIT = {"train": 3, "val": 3, "test": 2}
FAULT_DAY_FRACTION = 0.6
MISSING_PARTITIONS_PER_ARRANGEMENT = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest() -> dict:
    return json.loads((DATA_DIR / "manifest.json").read_text())


def _rows(arr: harness.Arrangement, split: str, faulted_only: bool, severity_by_day: dict) -> pd.DataFrame:
    df = harness.extract_day_features(arr)
    if faulted_only:
        df = df[df["is_fault"] == 1]
    df = df.copy()
    df["split"] = split
    df["pickup_date"] = df["pickup_date_str"]
    df["severity"] = df["pickup_date"].map(lambda d: severity_by_day.get(d, {}).get("fraction", 1.0)
                                            if d in severity_by_day else 0.0)
    df["expected_feature"] = df["fault_type"].map(EXPECTED_FEATURE)
    df["legit_event"] = df["pickup_date"].map(LEGIT_EVENTS).fillna("")
    df["day_of_week"] = pd.to_datetime(df["pickup_date"]).dt.dayofweek
    keep = ["split", "arrangement", "pickup_date", *FEATURE_COLUMNS, "is_fault", "fault_type", "severity",
            "expected_feature", "ground_truth_asset", "ground_truth_mechanism", "legit_event", "day_of_week"]
    return df[keep]


def build_split(split: str, days: list[str], json_path: str, work_dir: str, seed: int):
    frames, missing = [], []
    clean = harness.build_clean_arrangement(f"{split}_clean", json_path, days, work_dir)
    log(f"[{split}] clean arrangement: {len(days)} days, dbt build {clean.build_seconds:.1f}s")
    frames.append(_rows(clean, split, faulted_only=False, severity_by_day={}))
    rng = random.Random(seed)

    for a in range(ARRANGEMENTS_PER_SPLIT[split]):
        chosen = rng.sample(days, k=round(FAULT_DAY_FRACTION * len(days)))
        types = (ROW_FAULT_TYPES * (len(chosen) // len(ROW_FAULT_TYPES) + 1))[: len(chosen)]
        rng.shuffle(types)
        assignments, severity_by_day = {}, {}
        for i, (day, ftype) in enumerate(zip(chosen, types)):
            if i < MISSING_PARTITIONS_PER_ARRANGEMENT:
                assignments[day] = "missing_partition"
                missing.append({"split": split, "arrangement": f"{split}_faults_{a}", "pickup_date": day})
                continue
            kwargs = rng.choice(SEVERITIES[ftype])
            assignments[day] = (ftype, kwargs)
            severity_by_day[day] = kwargs
        arr = harness.build_fault_arrangement(f"{split}_faults_{a}", json_path, days, assignments, work_dir)
        log(f"[{split}] fault arrangement {a}: {len(assignments)} faulted days, dbt build {arr.build_seconds:.1f}s")
        frames.append(_rows(arr, split, faulted_only=True, severity_by_day=severity_by_day))
        # cross-check: completeness rule on the missing partitions (no row exists for ML to score)
        from adie.detection import completeness

        flagged = {s.raw["pickup_date"] for s in completeness.detect(arr.db_path)}
        for m in missing:
            if m["arrangement"] == arr.name:
                m["completeness_detected"] = int(m["pickup_date"] in flagged)
        Path(arr.db_path).unlink(missing_ok=True)
    Path(clean.db_path).unlink(missing_ok=True)
    return pd.concat(frames, ignore_index=True), missing


def natural_anomaly_days(json_path: str) -> dict[str, str]:
    """Real data-quality incidents already present in the PUBLISHED source
    (found while building this dataset, e.g. 2023-09-22..24 carry ~4% of a
    normal day's trips in Socrata itself). Flagged by a fixed rule on the
    raw rows -- never by any detector's output:
      * source_volume_gap: day has < 50% of the split's median trip count
        and is not one of the pre-registered LEGIT_EVENTS (Christmas etc.);
      * impossible_trip: a single trip_distance > 1,000 miles.
    These days are neither 'clean' (they are excluded from clean training,
    calibration, and false-positive rates) nor injected faults; detection
    of them is reported separately."""
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            "select cast(tpep_pickup_datetime::date as varchar) d, count(*) n, max(trip_distance::double) mx "
            f"from read_json_auto('{Path(json_path).as_posix()}') group by 1").fetchall()
    finally:
        con.close()
    med = float(pd.Series([r[1] for r in rows]).median())
    out = {}
    for d, n, mx in rows:
        if n < 0.5 * med and d not in LEGIT_EVENTS:
            out[d] = "source_volume_gap"
        elif mx is not None and mx > 1000:
            out[d] = "impossible_trip"
    return out


def _annotate(features: pd.DataFrame) -> pd.DataFrame:
    manifest = load_manifest()
    nat = {}
    for split in ("train", "val", "test"):
        nat.update(natural_anomaly_days(manifest["splits"][split]["path"]))
    features = features.copy()
    features["natural_anomaly"] = features["pickup_date"].map(nat).fillna("")
    return features


def build_or_load(rebuild: bool = False, seed: int = 2023) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_path = DATA_DIR / "missing_partitions.csv"
    if FEATURES_CACHE.exists() and missing_path.exists() and not rebuild:
        return _annotate(pd.read_parquet(FEATURES_CACHE)), pd.read_csv(missing_path)

    manifest = load_manifest()
    work_dir = harness.make_workdir("hybrid")
    try:
        frames, missing = [], []
        for k, split in enumerate(["train", "val", "test"]):
            info = manifest["splits"][split]
            df, miss = build_split(split, info["days"], info["path"], work_dir, seed + k)
            frames.append(df)
            missing += miss
    finally:
        harness.cleanup_workdir(work_dir)
    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(FEATURES_CACHE, index=False)
    missing_df = pd.DataFrame(missing)
    missing_df.to_csv(missing_path, index=False)
    return _annotate(features), missing_df
