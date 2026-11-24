# Autonomous Data Infrastructure Engine (ADIE)

A working vertical slice of a self-healing data pipeline: real NYC taxi
trip data flows through dbt + Dagster, two independent detectors (a dbt/
Elementary-style rule check and an sklearn IsolationForest) watch the daily
aggregate, a lineage-constrained root-cause scorer ranks which upstream
asset is responsible, and a LangGraph agent proposes a repair that must
pass sandbox verification -- and get human approval -- before it ever
touches the real warehouse file.

```
NYC Taxi (Socrata) -> raw.trips (DuckDB)
    -> dbt: stg_trips, stg_zones -> trips_enriched -> fct_trips_daily
        -> detection: dbt tests / Elementary + IsolationForest + completeness check
            -> lineage-constrained RCA (adie/rca)
                -> LangGraph agent: propose -> [human approval] -> sandbox verify -> apply
```

## What's reused vs. built here

| Plane | Reused | Built here |
|---|---|---|
| Substrate | Dagster, dbt-core, dbt-duckdb, DuckDB | one asset to load the raw JSON feed |
| Detection (rules) | dbt tests, `dbt-labs/dbt_utils`, `elementary-data/elementary` | schema/test YAML |
| Detection (ML) | scikit-learn `IsolationForest`, `RandomForestClassifier`, `xgboost.XGBClassifier` | the daily-metrics adapter, per-feature z-score decomposition, and the offline fit/predict research harness in `adie/eval/` |
| Lineage | dbt's own manifest, `networkx` | the manifest -> graph loader + cone queries |
| RCA | PyRCA's graph-proximity *idea* (not the library -- see `adie/rca/scorer.py` docstring); `RandomForestClassifier` for the no-lineage ML baseline | the cone-constrained scorer + metric->mechanism table; `adie/eval/rca_baselines.py`'s two comparison baselines |
| Agent | LangGraph, `langchain-anthropic` (`claude-opus-5`) | the graph wiring, repair catalog, sandbox verifier |
| Control | Python stdlib `sqlite3` | the incident/audit log |
| Dashboard | Dagster's own UI (assets, checks, run history, job launchpad) | nothing -- no custom dashboard was needed |
| Research harness | pandas, scikit-learn metrics, matplotlib | `adie/eval/`, `scripts/run_research_experiments.py`, `scripts/generate_figures.py` |

**Research deliverables** (Phase 1-4 experiments, real NYC taxi data, real train/test date separation — not fabricated): see
[`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) (methodology, results, discussion, limitations),
[`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) (quick-reference tables + figures),
[`ML_RESULTS.csv`](ML_RESULTS.csv) (raw per-row predictions),
and `experiments/results/figures/*.png`. [`DEMO_GUIDE.md`](DEMO_GUIDE.md) has the exact commands to reproduce or present any of this.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# dbt packages (dbt_utils, elementary)
.venv/bin/dbt deps --project-dir dbt_project

# dbt looks for ~/.dbt/profiles.yml by convention; this repo's profile reads
# the warehouse path from $ADIE_DUCKDB_PATH, so point it wherever you like:
mkdir -p ~/.dbt
cat > ~/.dbt/profiles.yml <<'PROFILE'
adie:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('ADIE_DUCKDB_PATH') }}"
      threads: 4
PROFILE

export ADIE_DUCKDB_PATH="$PWD/data/warehouse/adie.duckdb"
```

### Get the data

The TLC's own CloudFront bucket sits behind an AWS WAF bot challenge that a
plain HTTP client can't pass, and this project does not attempt to bypass
it. Instead `scripts/fetch_data.py` pulls the same canonical TLC schema
from NYC's official Socrata open-data API (`data.cityofnewyork.us`,
dataset `4b4i-vvec`, "2023 Yellow Taxi Trip Data") -- real trips, just a
different legitimate distribution point -- plus the small zone-lookup
reference table from the CloudFront bucket's `/misc/` path (not gated):

```bash
.venv/bin/python scripts/fetch_data.py   # ~14 days x 4,000 trips/day by default
```

A flat sample over one month lands almost entirely on day one (NYC yellow
taxi does >100k trips/day), so the script pulls a fixed number of rows
*per day* across a date range instead -- that's what gives
`fct_trips_daily` fourteen comparable rows for the detectors to reason
about, rather than one huge day and thirteen empty ones.

### Build the warehouse

```bash
.venv/bin/python -c "
import duckdb
con = duckdb.connect('data/warehouse/adie.duckdb')
con.execute('create schema if not exists raw')
con.execute(\"create or replace table raw.trips as select * from read_json_auto('data/raw/yellow_tripdata_2023-01_sample.json', union_by_name=true)\")
con.close()
"
.venv/bin/dbt build --project-dir dbt_project --profiles-dir ~/.dbt
```

(Dagster does this same load automatically via the `raw_trips` asset --
the manual step above is only needed if you want a warehouse before the
first Dagster materialization.)

## Run it

**Dagster UI** (assets, check results, run history -- this *is* the
monitoring dashboard, nothing custom was built):

```bash
export DAGSTER_HOME="$PWD/.dagster_home"
.venv/bin/dagster dev -m dagster_project.definitions
```

Open http://localhost:3000. Open any asset (e.g. `fct_trips_daily`) and use
its **Materialize** button, then its **Checks** tab to watch
`isolation_forest_anomaly_check` / `calendar_completeness_check` / every
dbt test render as a real Dagster asset check, with Dagster's own
metadata-driven plots.

> **Known issue, not fixed by design** (Phase 6 of this project explicitly
> deprioritized cosmetic dashboard work): the asset-graph *picture* --
> Catalog's graph view, the per-asset Lineage tab, the Jobs graph page --
> hangs on a loading spinner in some browser contexts. Confirmed via direct
> GraphQL query that the backend data is correct; this is a frontend
> rendering bug, not a data problem. Every list/detail page (an asset's
> Overview/Checks/Events tabs, Runs) renders correctly and is the
> recommended way to inspect the system through the UI.

**Inject a fault, then investigate + repair it:**

```bash
.venv/bin/python -c "
from adie.faults.inject import apply
apply('data/warehouse/adie.duckdb', 'missing_partition', day='2023-01-09')
"
.venv/bin/dbt build --project-dir dbt_project --profiles-dir ~/.dbt --select trips_enriched fct_trips_daily

# Trigger the agent from Dagster's own job launchpad (or the CLI below) --
# it investigates, ranks hypotheses, proposes a repair, and PAUSES for
# approval. The run log prints the incident id and the exact resume call:
.venv/bin/dagster job execute -m dagster_project.definitions -j investigate_and_repair_job
```

Approve or reject the paused incident (this is the human-in-the-loop gate
-- LangGraph's `interrupt()` persists the run via a SQLite checkpointer, so
this can happen minutes or days later, in a different process):

```python
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from adie.agent.graph import make_graph

with SqliteSaver.from_conn_string("data/warehouse/agent_checkpoints.sqlite") as cp:
    graph = make_graph(
        duckdb_path="data/warehouse/adie.duckdb", dbt_project_dir="dbt_project",
        manifest_path="dbt_project/target/manifest.json", checkpointer=cp,
        raw_json_path="data/raw/yellow_tripdata_2023-01_sample.json",
        incident_db_path="data/warehouse/incidents.sqlite",
    )
    config = {"configurable": {"thread_id": "<incident-id-from-the-log>"}}
    final = graph.invoke(Command(resume=True), config=config)   # or resume=False to reject
    print(final["final_status"])
```

Approving runs the proposed repair against a **throwaway clone** of the
warehouse first (`adie/agent/sandbox.py`); it only touches the real
`adie.duckdb` file if that sandboxed run passes. Rejecting leaves
production untouched.

### Five fault types, one safety behavior worth noting

`adie/faults/inject.py` covers all five required fault types: missing
partitions, duplicate records, schema changes, join errors, and incorrect
transformations. The first four are data-layer faults the agent can
safely auto-repair (it re-ingests the affected day from the original
source file, or deduplicates). The fifth -- a bug in `stg_trips.sql`
itself -- is **not** auto-repaired: the RCA scorer correctly attributes it
to `transformation_defect`, and the agent's repair catalog has no safe
data patch for that mechanism, so it proposes `no_op` and escalates to a
human instead of guessing at a code fix. See `adie/agent/tools.py`'s
`REPAIR_CATALOG_DESCRIPTION`.

### With a live LLM

Set `ANTHROPIC_API_KEY` and the `propose_repair` node calls
`claude-opus-5` (via `langchain-anthropic`, structured output) instead of
the deterministic fallback used in every test and demo run above. Every
`Action` and the graph state carry `llm_used: bool` so a run never silently
claims to be LLM-driven when it wasn't.

## Tests

```bash
.venv/bin/dbt build --project-dir dbt_project --profiles-dir ~/.dbt   # generates target/manifest.json, needed by most tests
.venv/bin/pytest -q
```

`tests/test_rca_eval.py` is the falsifiable one: it injects each fault type
into a throwaway copy of the warehouse and asserts the RCA ranker actually
finds the ground-truth asset, not just that it runs without error.

## Research experiments

The 36 pytest tests above check correctness. A separate, larger harness
(`adie/eval/`) produced the numbers in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md)
and [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md): 39 independently-built
real-data warehouses (28 held-out TRAIN days, 14 held-out TEST days, Feb vs.
March 2023, strictly disjoint dates), comparing Isolation Forest / Random
Forest / XGBoost / a statistical baseline on anomaly detection, and three
root-cause-analysis methods, then running the full agent repair loop on
every fault case. Reproduce with:

```bash
.venv/bin/python scripts/fetch_experiment_data.py        # ~1 min, one-time
.venv/bin/python scripts/run_research_experiments.py     # ~15-17 min: 39 real dbt builds + 2 model fits + 16 agent repairs
.venv/bin/python scripts/generate_figures.py
```

See [`DEMO_GUIDE.md`](DEMO_GUIDE.md) §8 for the full command block with environment variables.

## Docker

```bash
# fetch real data on the host first -- it's gitignored, so nothing is
# baked into the image unless it exists in data/ before you build:
.venv/bin/python scripts/fetch_data.py

docker compose up --build
```

Starts `dagster dev` on port 3000 with the repo's `data/` directory
mounted (so the container sees the same `data/raw/*.json` fetched above,
and persists `data/warehouse/adie.duckdb` back to the host). Set
`ANTHROPIC_API_KEY` in the environment before `docker compose up` to get
live LLM proposals instead of the deterministic fallback.

**Not verified in this environment** -- the sandbox this project was built
in has no `docker` binary, so the Dockerfile and compose file were written
against the same install steps verified above (and reviewed for an
obvious `.dockerignore` miss -- `.venv`/`.git`/dbt's `target/` are now
excluded from the build context) but have not themselves been built or
run. Verify on a machine with Docker before relying on it.

## Project layout

```
adie/
  contracts.py         Signal -> Hypothesis -> Action -> Verification envelope; validate_day() SQL-injection guard
  lineage/graph.py      dbt manifest -> networkx graph, upstream/downstream cone queries
  detection/            IsolationForest adapter + calendar-completeness rule check
  rca/scorer.py         cone-constrained hypothesis ranking
  agent/                LangGraph graph, repair catalog, sandbox verifier
  control/              SQLite incident/audit log
  faults/inject.py      the 5 required fault types
  eval/                 research-experiment harness: arrangement builder, the 4 ML models,
                         the 2 RCA comparison baselines, Wilson-interval stats helper
dagster_project/        Dagster Definitions: dbt assets, detection checks, the agent job
dbt_project/             dbt models (staging/marts), seeds, Elementary + dbt_utils packages
scripts/
  fetch_data.py          reproducible pull of real NYC TLC data from Socrata (14-day demo dataset)
  fetch_experiment_data.py  pulls the 28-train/14-test disjoint research dataset
  run_research_experiments.py  the Phase 1/2/4 experiment runner -- produces experiments/results/
  generate_figures.py   renders the PNGs in experiments/results/figures/ from that run's output
experiments/
  data/                 fetched research-dataset JSON, ~30MB (gitignored -- regenerate via fetch_experiment_data.py)
  results/               committed: rca_results.csv / repair_results.csv / summary.json / figures/*.png
                         (gitignored: the run's own incident sqlite + log -- regenerate the whole
                         directory via run_research_experiments.py + generate_figures.py)
tests/                  pytest suite, including the RCA eval harness
docker/, docker-compose.yml
RESEARCH_REPORT.md, EVALUATION_REPORT.md, ML_RESULTS.csv, DEMO_GUIDE.md   research deliverables (see above)
```
