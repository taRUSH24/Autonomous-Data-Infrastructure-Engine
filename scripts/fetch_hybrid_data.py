#!/usr/bin/env python3
"""
Pulls the dataset for the hybrid-ensemble experiments: a whole calendar
year (2023) of REAL NYC yellow-taxi trips, uniformly sampled within each
day (~1/60 of all trips, see scripts/fetch_data.py:fetch_day_uniform), and
splits it CHRONOLOGICALLY:

    live demo window  2023-01-01 .. 2023-01-14   (data/raw demo file; never used for fitting)
    TRAIN             2023-01-15 .. 2023-06-30   (models are fit here)
    VALIDATION        2023-07-01 .. 2023-08-31   (thresholds / fusion weights calibrated here)
    TEST              2023-09-01 .. 2023-12-31   (reported once, never used for any choice)

Why: the previous experiments had 28 + 14 independent days, with each
day's sample being the first 1,500 trips after midnight -- so trip_count
was constant by construction and only early-morning trips were seen.
365 independent real days covering the full 24h is more real data, not
synthetic duplication, and the TEST window naturally contains legitimate
distribution changes (Labor Day, the 2023-09-29 flash flood, Thanksgiving,
Christmas, New Year's Eve) to measure false alarms against.

Per-day responses are cached in experiments/data/hybrid/days/ so the
fetch is resumable.

Run: .venv/bin/python scripts/fetch_hybrid_data.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_data import fetch_day_uniform  # noqa: E402

SPLITS = {
    "demo": ("2023-01-01", "2023-01-14"),
    "train": ("2023-01-15", "2023-06-30"),
    "val": ("2023-07-01", "2023-08-31"),
    "test": ("2023-09-01", "2023-12-31"),
}


def days_between(a: str, b: str) -> list[datetime.date]:
    start, end = datetime.date.fromisoformat(a), datetime.date.fromisoformat(b)
    return [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="experiments/data/hybrid")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--demo-out", default="data/raw/yellow_tripdata_2023-01_sample.json",
                    help="where to also write the demo-window rows (the live Dagster pipeline's input)")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache = out_dir / "days"
    cache.mkdir(parents=True, exist_ok=True)

    all_days = [d for (a, b) in SPLITS.values() for d in days_between(a, b)]

    def get(day: datetime.date) -> int:
        path = cache / f"{day}.json"
        if path.exists():
            return len(json.loads(path.read_text()))
        rows = fetch_day_uniform(day)
        path.write_text(json.dumps(rows))
        print(f"  {day}: {len(rows)} rows", flush=True)
        return len(rows)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        counts = dict(zip(all_days, ex.map(get, all_days)))

    manifest = {"sampling": "uniform: pickup second == 7 (~1/60 of all trips)", "splits": {}}
    for split, (a, b) in SPLITS.items():
        days = days_between(a, b)
        rows = []
        for d in days:
            rows.extend(json.loads((cache / f"{d}.json").read_text()))
        if split == "demo":
            if args.skip_demo:
                continue
            target = Path(args.demo_out)
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = out_dir / f"{split}.json"
        target.write_text(json.dumps(rows))
        manifest["splits"][split] = {"start": a, "end": b, "n_days": len(days), "n_rows": len(rows),
                                     "path": str(target), "days": [str(d) for d in days]}
        print(f"{split}: {len(days)} days, {len(rows)} rows -> {target}")

    empty = [str(d) for d, n in counts.items() if n == 0]
    manifest["empty_days"] = empty
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"empty days: {empty or 'none'}")


if __name__ == "__main__":
    main()
