#!/usr/bin/env python3
"""
Pulls a larger, date-disjoint real NYC taxi dataset for the ML research
harness (adie/eval): a TRAIN pool and a TEST pool of calendar days that
never overlap, so supervised models are evaluated on genuinely unseen
dates -- not just a random row-level split of the same days.

Same Socrata source as scripts/fetch_data.py (see that file's docstring for
why: the TLC's own CloudFront bucket is behind a bot-detection challenge
this project does not attempt to bypass).
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

SOCRATA_BASE = "https://data.cityofnewyork.us/resource/4b4i-vvec.json"
USER_AGENT = "adie-research-harness/0.1 (NYC taxi anomaly-detection research project)"


def fetch_day(day: datetime.date, limit: int, retries: int = 4) -> list[dict]:
    next_day = day + datetime.timedelta(days=1)
    where = f"tpep_pickup_datetime between '{day}T00:00:00' and '{next_day}T00:00:00'"
    params = {"$limit": limit, "$where": where, "$order": "tpep_pickup_datetime"}
    url = SOCRATA_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (503, 429) and attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    {day}: HTTP {e.code}, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-02-01", help="first day to pull (YYYY-MM-DD)")
    ap.add_argument("--train-days", type=int, default=28)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--per-day", type=int, default=1500, help="rows per day (kept modest: dbt rebuilds many times in the harness)")
    ap.add_argument("--out-dir", default="experiments/data")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.date.fromisoformat(args.start)
    total_days = args.train_days + args.test_days
    all_days = [start + datetime.timedelta(days=i) for i in range(total_days)]
    train_pool = all_days[: args.train_days]
    test_pool = all_days[args.train_days :]

    def pull(days: list[datetime.date], label: str) -> None:
        rows: list[dict] = []
        for d in days:
            batch = fetch_day(d, args.per_day)
            rows.extend(batch)
            print(f"  [{label}] {d}: +{len(batch)} rows (total {len(rows)})")
            time.sleep(0.2)
        path = out_dir / f"{label}.json"
        path.write_text(json.dumps(rows))
        print(f"wrote {len(rows)} rows -> {path}")

    print(f"TRAIN pool: {train_pool[0]} .. {train_pool[-1]} ({len(train_pool)} days)")
    pull(train_pool, "train_pool")

    print(f"\nTEST pool: {test_pool[0]} .. {test_pool[-1]} ({len(test_pool)} days)")
    pull(test_pool, "test_pool")

    (out_dir / "manifest.json").write_text(json.dumps({
        "train_days": [str(d) for d in train_pool],
        "test_days": [str(d) for d in test_pool],
        "per_day": args.per_day,
    }, indent=2))
    print(f"\nTRAIN and TEST day pools are disjoint: {set(train_pool).isdisjoint(test_pool)}")


if __name__ == "__main__":
    main()
