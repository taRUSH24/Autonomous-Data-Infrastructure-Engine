#!/usr/bin/env python3
"""
Pulls a real, stratified sample of NYC yellow-taxi trip records from the
city's official open-data portal (not a static file this repo ships).

NOTE on source choice: the NYC TLC's own CloudFront bucket
(d37ci6vzurychx.cloudfront.net) sits behind an AWS WAF bot challenge that a
plain HTTP client cannot pass, and this project does not attempt to -- see
CLAUDE.md / docs/architecture for the reasoning. Socrata
(data.cityofnewyork.us) is NYC's official open-data API serving the exact
same TLC dataset (catalog id 4b4i-vvec, "2023 Yellow Taxi Trip Data") with
no such gate, so that's the source used here.

Pulls `--per-day` rows for each of `--days` consecutive days starting at
`--start`, so every day has comparable volume -- a flat single-month pull
lands almost entirely on day one (NYC yellow taxi does >100k trips/day).
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
# This one small reference file (265 rows) IS reachable from the TLC's own
# CloudFront bucket with a plain GET + User-Agent header -- only the large
# trip-data parquet endpoints sit behind the WAF bot challenge mentioned
# above, not this one.
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
USER_AGENT = "adie-vertical-slice/0.1 (NYC taxi anomaly-detection demo project)"


def fetch_zone_lookup(out_path: Path) -> None:
    req = urllib.request.Request(ZONE_LOOKUP_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        out_path.write_bytes(resp.read())


def fetch_day(day: datetime.date, limit: int) -> list[dict]:
    next_day = day + datetime.timedelta(days=1)
    where = f"tpep_pickup_datetime between '{day}T00:00:00' and '{next_day}T00:00:00'"
    params = {"$limit": limit, "$where": where, "$order": "tpep_pickup_datetime"}
    url = SOCRATA_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch_day_uniform(day: datetime.date, second: int = 7, limit: int = 50000, retries: int = 5) -> list[dict]:
    """Uniform ~1/60 sample of *every* trip that day: keeps trips whose
    pickup timestamp has seconds == `second`. Unlike fetch_day() (first N
    trips ordered by time, i.e. only ~00:00-06:00), this covers all 24h and
    lets trip_count vary with real demand (weekday/weekend/holiday/season),
    which is what a volume-based detector actually needs to learn."""
    next_day = day + datetime.timedelta(days=1)
    where = (
        f"tpep_pickup_datetime >= '{day}T00:00:00' and tpep_pickup_datetime < '{next_day}T00:00:00' "
        f"and date_extract_ss(tpep_pickup_datetime) = {int(second)}"
    )
    params = {"$limit": limit, "$where": where, "$order": "tpep_pickup_datetime"}
    url = SOCRATA_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    {day}: {e}, retrying in {wait}s")
            time.sleep(wait)
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2023-01-01", help="first day to pull (YYYY-MM-DD)")
    ap.add_argument("--days", type=int, default=14, help="number of consecutive days")
    ap.add_argument("--per-day", type=int, default=4000, help="rows to pull per day (--sampling first-n only)")
    ap.add_argument(
        "--sampling", choices=["first-n", "uniform"], default="first-n",
        help="first-n: first --per-day trips of each day by pickup time (original behaviour, "
             "covers ~00:00-06:00 only); uniform: ~1/60 sample spread across all 24h "
             "(what the hybrid ensemble's pretrained models expect)",
    )
    ap.add_argument("--out", default="data/raw/yellow_tripdata_2023-01_sample.json")
    ap.add_argument("--zone-lookup-out", default="dbt_project/seeds/taxi_zone_lookup.csv")
    args = ap.parse_args()

    zone_path = Path(args.zone_lookup_out)
    zone_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching zone lookup -> {zone_path}")
    fetch_zone_lookup(zone_path)

    start = datetime.date.fromisoformat(args.start)
    rows: list[dict] = []
    for i in range(args.days):
        day = start + datetime.timedelta(days=i)
        batch = fetch_day_uniform(day) if args.sampling == "uniform" else fetch_day(day, args.per_day)
        rows.extend(batch)
        print(f"  {day}: +{len(batch)} rows (total {len(rows)})")
        time.sleep(0.25)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
