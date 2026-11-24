# Demo Guide

Exact commands to demonstrate the Autonomous Data Infrastructure Engine end to end. Run from the repo root on Ubuntu with the virtualenv active:

```bash
python3 -m venv .venv          # if not already created
.venv/bin/pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
export ADIE_DUCKDB_PATH="$PWD/data/warehouse/adie.duckdb"
export DAGSTER_HOME="$PWD/.dagster_home"
export DBT_PROFILES_DIR="$HOME/.dbt"
```

`~/.dbt/profiles.yml` must exist first — see README.md "Install" if this is a fresh machine.

## 1. Show the pipeline is real: run the tests

```bash
dbt build --project-dir dbt_project --profiles-dir ~/.dbt   # builds the warehouse + generates target/manifest.json, which most tests need
pytest -q
```

Expect: `36 passed`. Takes ~5-6 minutes (most of it is dbt subprocess rebuilds inside the test suite, not slow tests).

## 2. Start the dashboard

```bash
dagster dev -m dagster_project.definitions
```

Open **http://localhost:3000**. Click **Catalog** in the left nav to see every asset (dbt models + the two custom checks). Click **Runs** for history.

> Known issue, not worth your time debugging live: the asset-graph/lineage *picture* (Catalog page's graph view, the Lineage tab) hangs on a spinner in some browser contexts — it's a frontend rendering bug, confirmed via direct GraphQL query that the backend data is fine. Every list/detail page (an asset's Overview/Checks/Events tabs, Runs) renders correctly. If asked, say this plainly rather than trying to fix it on the spot.

## 3. Materialize the pipeline through the dashboard

In the UI: open the `fct_trips_daily` asset → click **Materialize** (top right). Watch it run in **Runs**. This executes the real `dbt build` command against real NYC taxi data pulled from NYC's official Socrata open-data API.

## 4. Inject a fault and show detection

```bash
python -c "
from adie.faults.inject import apply
apply('data/warehouse/adie.duckdb', 'join_error', day='2023-01-06')
"
```

Back in the UI: open `trips_enriched` → **Materialize**, then `fct_trips_daily` → **Materialize** (both are needed — `trips_enriched` must be rebuilt first for the corrupted join to propagate). Open `fct_trips_daily` → **Checks** tab: `dbt_utils_accepted_range_..._zone_count__50` and `isolation_forest_anomaly_check` now show **Failed**. Click `isolation_forest_anomaly_check` → `[Show JSON]` on `signals` to show the exact day and metric that fired.

## 5. Show root-cause analysis

```bash
python -c "
from adie.detection import isolation_forest, completeness
from adie.lineage.graph import LineageGraph
from adie.rca.scorer import rank_hypotheses

signals = isolation_forest.detect('data/warehouse/adie.duckdb') + completeness.detect('data/warehouse/adie.duckdb')
lg = LineageGraph.from_manifest('dbt_project/target/manifest.json')
hyps = rank_hypotheses(lg, signals)
for h in hyps[:3]:
    print(f'{h.score:5.3f}  {h.asset_key:16s} mechanism={h.mechanism}')
"
```

Top hypothesis should be `trips_enriched` / `join_defect` — exactly the fault injected in step 4.

## 6. Run the agentic repair workflow

```bash
dagster job execute -m dagster_project.definitions -j investigate_and_repair_job
```

This prints the proposed repair, its rationale, `llm_used` (True only if `ANTHROPIC_API_KEY` is set — see README's note on this), and the exact `thread_id` to resume with. Approve it:

```bash
python -c "
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from adie.agent.graph import make_graph

with SqliteSaver.from_conn_string('data/warehouse/agent_checkpoints.sqlite') as cp:
    graph = make_graph(
        duckdb_path='data/warehouse/adie.duckdb', dbt_project_dir='dbt_project',
        manifest_path='dbt_project/target/manifest.json', checkpointer=cp,
        raw_json_path='data/raw/yellow_tripdata_2023-01_sample.json',
        incident_db_path='data/warehouse/incidents.sqlite',
    )
    config = {'configurable': {'thread_id': '<PASTE THE INCIDENT ID FROM THE LOG ABOVE>'}}
    final = graph.invoke(Command(resume=True), config=config)
    print(final['final_status'])
"
```

Re-materialize `fct_trips_daily` in the UI once more: the check that was failing is back to **Succeeded**.

## 7. Show the incident audit trail

```bash
python -c "
from adie.control.incident_store import IncidentStore
store = IncidentStore('data/warehouse/incidents.sqlite')
for ev in store.timeline('<the same incident id>'):
    print(ev['created_at'], ev['event_type'])
"
```

Shows: evidence gathered → repair proposed → approval decision → sandbox verification → final status, with real timestamps.

## 8. Run the full research experiment suite (optional — this is what produced ML_RESULTS.csv / EVALUATION_REPORT.md)

```bash
python scripts/fetch_experiment_data.py          # only needed once; ~1 min
python scripts/run_research_experiments.py       # ~12-15 min: builds ~39 independent dbt warehouses, trains/evaluates 4 ML models + 3 RCA methods, runs the agent repair loop
```

Outputs land in `experiments/results/`: `ml_results.csv`, `rca_results.csv`, `repair_results.csv`, `false_alarm_results.csv`, `summary.json`. `RESEARCH_REPORT.md` and `EVALUATION_REPORT.md` were written from this run's actual output — not fabricated.

## If something doesn't start

- `dbt build` errors about a missing profile → create `~/.dbt/profiles.yml` per README.md.
- Port 3000 already in use → `dagster dev -p 3001 ...` and adjust the URL.
- `ANTHROPIC_API_KEY` not set → expected in this environment; the agent uses its deterministic fallback and labels every run `llm_used: False`. This is disclosed, not hidden — see RESEARCH_REPORT.md's limitations section.
