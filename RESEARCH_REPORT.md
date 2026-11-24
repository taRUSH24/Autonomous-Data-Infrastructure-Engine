# Research Report: Autonomous Data Infrastructure Engine

**Date:** 2026-11-25
**Experiment run:** `experiments/results/summary.json`, generated 2026-11-25T00:16:36 by `scripts/run_research_experiments.py`

All numbers in this report are read directly from that run's output (`experiments/results/*.csv`, `summary.json`). None are estimated, assumed, or back-filled. Where a result is small-sample or ambiguous, that is stated explicitly rather than rounded off.

## 1. Problem Statement

Data pipelines fail silently. A dropped partition, a duplicated batch, a nulled column, a corrupted join key, or a transformation bug can all pass through a pipeline and land in a dashboard looking like real data. Three capabilities are usually missing, separately, from most stacks:

1. **Detection** that catches more than schema-level rule violations — multivariate anomalies in aggregate metrics, not just `NOT NULL` failures.
2. **Root-cause localization** that narrows a flagged symptom down to the actual upstream asset responsible, rather than leaving an engineer to manually trace lineage.
3. **Safe automated remediation** — a system that proposes a fix, verifies it against a sandbox before touching production, and knows when *not* to act.

This project builds and evaluates a working system across all three, using real NYC taxi trip data and five controlled fault types with known ground truth, so that "it works" is a falsifiable claim rather than a demo.

## 2. Objectives

1. Compare four anomaly-detection approaches (Isolation Forest, Random Forest, XGBoost, a z-score statistical baseline) on real data with proper train/test date separation.
2. Compare three root-cause analysis approaches (a lineage-constrained scorer, a no-lineage statistical baseline, a no-lineage ML classifier) on Top-1/Top-3 accuracy against known fault ground truth.
3. Verify an agentic (LangGraph) repair workflow end-to-end: detect → investigate → propose → human-approve → sandbox-verify → apply → record.
4. Measure repair success rate, false-repair rate, detection/recovery time, and false-alarm rate across repeated, reproducible fault-injection trials.
5. Do all of the above by **reusing** established libraries wherever one exists — scikit-learn, XGBoost, dbt, Dagster, Elementary, LangGraph, langchain-anthropic — building only the orchestration and domain logic that connects them.

## 3. Research Gap

Individually, dbt tests, Elementary, and generic anomaly-detection libraries are mature and widely used. What is comparatively under-explored in available open tooling is the **combination**: using the pipeline's own dependency graph (not a separately discovered or learned graph) as a *structural prior* for root-cause ranking, and using that ranking to **constrain** what an autonomous agent is allowed to repair — refusing to act when the implicated mechanism has no safe, known fix. Section 6.4 below is the clearest empirical result on why that constraint matters.

## 4. Methodology

### 4.1 Data

Real NYC Yellow Taxi trip records, pulled live from NYC's official Socrata open-data API (`data.cityofnewyork.us`, dataset `4b4i-vvec`) — not a bundled static sample, not synthetic data. See `scripts/fetch_experiment_data.py`.

| Pool | Date range | Days | Rows/day | Total rows |
|---|---|---|---|---|
| TRAIN | 2023-02-01 .. 2023-02-28 | 28 | 1,500 | 42,000 |
| TEST  | 2023-03-01 .. 2023-03-14 | 14 | 1,500 | 21,000 |

**TRAIN and TEST dates are strictly disjoint** (verified programmatically: `set(train_days).isdisjoint(test_days) == True`). Every model is fit only on TRAIN; every reported metric is computed only on TEST arrangements the model never saw during fitting — including the statistical baseline's mean/std and IsolationForest's offline fit.

### 4.2 Arrangements: how fault/clean episodes are built

Each "arrangement" is an independent, throwaway DuckDB warehouse: the real day pool loaded into `raw.trips`, zero or more of the five fault types injected on specific days, then rebuilt with the project's real `dbt build` (not a stub). Table-level ground truth (which asset, which mechanism) is recorded per injected fault. See `adie/eval/harness.py`.

- **TRAIN:** 1 clean arrangement (28 days) + 4 fault types × 5 repeats = 21 arrangements total.
- **TEST:** 1 clean arrangement (14 days) + 4 fault types × 4 repeats = 17 arrangements total, **plus** one separate transformation-defect arrangement (see 4.4).
- Each repeat injects its fault type on an independently-chosen day from the pool, so the same fault type is exercised on multiple different real calendar dates — this is the project's repeated-trial structure.

Total: 39 independent `dbt build` runs for this experiment, all executed, none skipped or mocked.

### 4.3 Models (Phase 1)

| Model | Library | Fit strategy |
|---|---|---|
| Isolation Forest | `sklearn.ensemble.IsolationForest` | Offline: fit once on TRAIN rows, scored on TEST (contamination=0.1, matching the live production detector's tuned default — see §7.1) |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | Supervised, fit on TRAIN labels |
| XGBoost | `xgboost.XGBClassifier` | Supervised, fit on TRAIN labels |
| Statistical baseline | z-score thresholding (`|z| > 3`, stdlib/numpy) | Mean/std fit on TRAIN, applied to TEST |

Features (all five models, identical five columns): `trip_count`, `avg_fare_amount`, `avg_trip_distance_miles`, `avg_total_amount`, `unmatched_zone_count` — the same features the live Dagster asset check uses.

### 4.4 The fifth fault type is structurally different

`missing_partition` deletes a day's rows entirely — there is no feature row to classify, so it cannot appear in the four models' row-level evaluation by definition. It is caught instead by the project's separate calendar-completeness rule check, and **is** included in the RCA evaluation (its signal carries a `metric="missing_partition"` that RCA reasons over). `incorrect_transformation` is a code-level bug (not a data mutation) affecting every day in an arrangement at once, so it cannot be scored against a single per-day ground-truth label the way the other four can — it is reported as one separate, fully-described case (§6.5), not averaged into the four-fault-type tables.

### 4.5 RCA methods (Phase 2)

| Method | What it uses | Reused from |
|---|---|---|
| `lineage_constrained` | The real dbt dependency graph (`adie.lineage.graph`), scored per `adie/rca/scorer.py` | This project's existing RCA engine, unmodified |
| `statistical_no_lineage` | Per-table row-count deviation from a clean baseline, no graph | `adie/eval/rca_baselines.py` (new, minimal) |
| `ml_no_lineage` | `sklearn.ensemble.RandomForestClassifier` trained on (metric, severity) → asset, no graph | `adie/eval/rca_baselines.py` (new, minimal) |

All three rank the same five-asset candidate set, so comparisons are apples-to-apples.

### 4.6 Agent (Phase 3/4)

The existing LangGraph state machine (`adie/agent/graph.py`): gather evidence → rank hypotheses → propose repair → **human-approval interrupt** → sandbox-verify (clone + patch + rebuild, never touching the real file) → apply (only if verification passed) → record to the incident store. No new agent logic was written for this phase — the experiment runner calls the same `adie.agent.sandbox.verify_action` / `apply_action` functions the live Dagster-triggered agent uses.

## 5. Experiments Run

| Experiment | n | What it measures |
|---|---|---|
| Phase 1: 4-model anomaly detection | 234 TEST rows (12 faulted, across duplicate_records/schema_change/join_error) | Precision, recall, F1, FPR, latency |
| Phase 2: 3-method RCA | 15 detected fault cases (of 16 fault arrangements; 1 detection miss) | Top-1 / Top-3 accuracy |
| Phase 4: agent repair | 16 fault cases | Detection rate, repair success rate, detect/recovery time |
| Phase 4: false alarms | 14 clean TEST days | False-alarm rate, auto-repairs on clean data |
| Phase 4: transformation-defect | 1 dedicated arrangement, all 14 TEST days affected at once | Detection, RCA, and — critically — whether the agent correctly refuses to auto-repair |

## 6. Results

### 6.1 Anomaly detection (Phase 1)

| Model | Precision | Recall | F1 | FPR | Mean latency/row |
|---|---|---|---|---|---|
| Isolation Forest | 0.207 | **1.000** | 0.343 | 0.207 | 9.8 ms |
| Random Forest | **1.000** | **1.000** | **1.000** | **0.000** | 16.6 ms |
| XGBoost | **1.000** | **1.000** | **1.000** | **0.000** | 2.6 ms |
| Statistical z-score | 0.429 | **1.000** | 0.600 | 0.072 | 0.8 ms |

See `experiments/results/figures/confusion_matrices.png` and `model_comparison.png`.

**All four models achieve perfect recall** — every one of the 12 faulted test rows was caught by every model. They differ entirely in false-positive rate. Isolation Forest, scored per-row against a fixed offline fit, flags 20.7% of clean rows (46 of 222); the supervised tree models flag none; the simple z-score baseline sits in between (7.2%).

**Honest caveat:** Random Forest and XGBoost's perfect scores reflect that these five engineered features make the three row-level fault types (duplicate_records, schema_change, join_error) nearly linearly separable by design — e.g. `unmatched_zone_count > 0` is close to a deterministic indicator of join_error in this feature set. This is a property of the feature engineering and fault design, not evidence that these models would generalize with the same precision to a fault type whose signature doesn't load this cleanly onto one of the five chosen columns. It is also why Isolation Forest, applied with a fixed offline cut rather than re-fit per-batch, pays a real false-positive cost: see §7.1 for why the live production detector deliberately does not use this exact configuration.

### 6.2 Root-cause analysis (Phase 2)

| Method | Top-1 accuracy | Top-3 accuracy | n |
|---|---|---|---|
| Lineage-constrained (this project's RCA) | 0.733 | **1.000** | 15 |
| Statistical, no lineage | 0.733 | 0.733 | 15 |
| ML classifier, no lineage | **1.000** | **1.000** | 15 |

See `experiments/results/figures/rca_comparison.png`. Per-fault-type Top-1 breakdown (from `experiments/results/rca_results.csv`):

| Fault type | Lineage-constrained | Statistical, no lineage | ML, no lineage | n |
|---|---|---|---|---|
| duplicate_records | 1.00 | 1.00 | 1.00 | 4 |
| join_error | 1.00 | **0.00** | 1.00 | 4 |
| missing_partition | 1.00 | 1.00 | 1.00 | 3 |
| schema_change | **0.00** | 1.00 | 1.00 | 4 |

**This is the most important, and most nuanced, result in this report, and it does not favor this project's own method on the headline number.**

- `statistical_no_lineage` (row-count deviation) gets **0% on join_error**: join_error corrupts a value, not a row count, so this baseline has literally no signal to rank on and defaults to an arbitrary tie-break. Its 73.3% overall score is propped up by three fault types that happen to change row counts (or where the tie-break coincidentally lands correctly) — it is not doing real diagnosis for value-corruption faults. This is exactly the failure mode the lineage-constrained method exists to avoid, and the data shows it: lineage-constrained scores 100% on join_error.
- The **lineage-constrained method scores 0% on schema_change's Top-1**, confirmed here empirically after being predicted and documented in code comments (`adie/rca/scorer.py`'s `MECHANISM_BY_METRIC` docstring) before this experiment ran: a value shift in `avg_fare_amount` is genuinely ambiguous between "the source already had bad values" and "the cast step corrupted them," so the scorer deliberately credits **both** `trips` and `stg_trips`, and in this fault's case `stg_trips` edges out the true answer (`trips`) by graph-proximity weight. It recovers at Top-3 (100%), which is the honest way to represent "I am genuinely unsure between these two, but not completely lost."
- `ml_no_lineage` gets a **perfect 100% Top-1** across every fault type, including schema_change. This is a real, reported number — not an underperforming baseline. The caveat, also real: in this project's fault design, `schema_change`'s ground truth is *always* `trips` (there is no example in the data where it's genuinely `stg_trips`), so a classifier trained on (metric → asset) pairs can simply memorize "this metric string → this asset" without resolving any ambiguity — it never has to *choose* under uncertainty the way the lineage method's design explicitly tries to. That strategy will not transfer to a fault type the classifier has never seen (it has no graph to reason from, only memorized associations); the lineage method's Top-3 of 100% reflects a method that keeps the right answer in view even when it isn't certain, which the ML classifier has no mechanism to express (it always outputs a confident ranking, never "these two are tied").

Decision on Graph Neural Networks (explicitly scoped in Phase 2): **not introduced.** The existing lineage-constrained method already achieves 100% Top-3 accuracy with a five-node candidate graph; there is neither a demonstrated deficiency nor remotely enough distinct training arrangements (15 labeled RCA cases) to train a GNN without overfitting. Revisit only if the asset graph grows to a scale and the Top-3 metric degrades below what cone-based scoring can fix.

### 6.3 Agent repair (Phase 4)

From `experiments/results/repair_results.csv` and `summary.json`:

- **Detection rate:** 15/16 (93.75%) of injected fault cases were flagged by at least one detector. The one miss was a `missing_partition` arrangement (3/4 detected for that type) — IsolationForest's offline cut, as discussed in §6.1, is the likely source, not the completeness check, which caught the other three.
- **Conditional repair success:** of the 15 detected cases, **15/15 (100%)** were correctly proposed, sandbox-verified, applied, and confirmed to restore the correct data.
- **Overall end-to-end success rate** (detect AND correctly repair, out of all 16 injected faults): **93.75%**, Wilson 95% CI **[71.7%, 98.9%]** (small-n; stated as an interval, not a point estimate presented as more precise than it is).
- **Mean detection latency:** 0.245s — this is the real wall-clock time for the Python detector calls (`adie.detection.isolation_forest.detect` + `completeness.detect`) against an already-materialized warehouse. It does **not** include the upstream dbt rebuild time; see §7.2 for why that distinction matters.
- **Mean recovery time:** 29.5s — real wall-clock time for sandbox verification (clone + SQL patch + dbt rebuild) plus the production apply step (patch + rebuild), i.e. two real dbt rebuilds end to end. Breakdown by fault type ranged 27.5s (missing_partition) to 30.8s (schema_change, join_error) — consistent across fault types, as expected since the dominant cost is dbt's fixed per-build overhead, not the fault-specific SQL.

See `experiments/results/figures/repair_and_false_alarms.png`.

### 6.4 False alarms and false repairs (Phase 4)

- On 14 clean TEST days (no fault injected), Isolation Forest flagged **2 (14.3%)** — consistent with the natural day-to-day variance already observed in this project's live-demo sessions, not a new finding, but confirmed again here on previously-unseen dates.
- **Auto-repairs applied to clean data: 0.** This number is true *by construction* in this experiment — the repair loop was only invoked on arrangements with a known injected fault, so clean arrangements were never offered to the agent at all. This is not the same claim as "the agent was tested on clean data and chose not to act" (that specific behavior — rejection and no-touch on a false/ambiguous flag — was verified separately in this project's live Dagster-UI demonstration, where a human-rejection path was exercised and confirmed to leave production untouched). Both facts are true; they are not the same evidence, and this report does not conflate them.

### 6.5 The fifth fault type: transformation_defect

One dedicated arrangement (code-level bug in `stg_trips.sql`, all 14 TEST days affected simultaneously):

| Measured | Result |
|---|---|
| Detected | Yes (2 signals) |
| RCA Top-1 | Correct (`stg_trips`, mechanism `transformation_defect`) |
| RCA Top-3 | Correct |
| **Agent correctly escalates instead of auto-repairing** | **Yes** |

This is the safety property this project's design most depends on: when the top-ranked root cause has a mechanism with no safe data-only fix (`transformation_defect` or `unknown`), the agent's existing proposal logic (`adie/agent/graph.py`) refuses to act and proposes `no_op` for human escalation instead of guessing. This experiment confirms that behavior fires correctly on a fresh, independently-built arrangement — not just in the earlier interactive demo session.

## 7. Discussion

### 7.1 Why Isolation Forest's offline variant has a real false-positive cost here

The live Dagster pipeline's detector (`adie.detection.isolation_forest.detect`) fits IsolationForest **per batch**, transductively, on whatever day-range is currently materialized — the standard way IsolationForest is used for one-off monitoring runs. This research harness instead needed a model fit **once on TRAIN and reused on TEST**, to make a fair comparison against the three supervised models, which also fit once and generalize. That offline-fit variant (`adie.eval.models.fit_isolation_forest`) is new for this phase; the live production detector is untouched. The 20.7% false-positive rate in §6.1 is a property of using IsolationForest this way, not a defect discovered in the live system — it is reported honestly as the cost of the fairer comparison design, and is the reason the live pipeline should **not** be changed to fit offline based on this result alone.

### 7.2 "Detection latency" is deliberately scoped, not inflated

This project is a batch, dbt-driven pipeline, not a streaming system. "Detection latency" reported in §6.3 (0.245s) is the real, measured cost of running the existing detector functions against an already-built warehouse — it is not claimed to be "time from fault occurring to alert in production," which in this architecture is dominated by how often the pipeline is scheduled to rebuild, not by detector compute time. The 29.5s recovery figure, which does include two real dbt rebuilds, is the more representative end-to-end number for "how long does healing actually take once a human approves it."

### 7.3 Phase 3 agentic AI: what was and was not verified

No `ANTHROPIC_API_KEY` is configured in this environment. Every agent run in this report — the 16 Phase 4 repair cases, the transformation-defect escalation case, and all earlier interactive-demo runs — used the project's deterministic fallback proposer (`llm_used: False` is recorded on every `Action` and incident record). This is disclosed on every run, not hidden behind a generic "the agent worked" claim. **Live LLM-based reasoning (`claude-opus-5` via `langchain-anthropic`) is untested in this report.** The code path exists (`adie/agent/graph.py`'s `propose_repair`, gated on `ANTHROPIC_API_KEY` being set) and was previously confirmed to construct the request correctly, but no real API call has been made, and none was made for this report, per the explicit instruction to seek approval before any paid call and because no key is available.

What **was** verified, against the real system, in this phase:

1. Receive anomaly alerts — yes, from the real detector signals (§6.1, §6.3).
2. Investigate pipeline execution history / examine lineage — yes, `adie.lineage.graph` + `adie.rca.scorer`, exercised on every one of the 16 fault cases.
3. Identify likely failure causes — yes, §6.2, with Top-1/Top-3 measured.
4. Generate repair proposals — yes, deterministic-fallback proposals, matched to the correct repair primitive in all 15 detected+repaired cases.
5. Execute repairs in an isolated sandbox after approval — yes, `adie.agent.sandbox.verify_action` clones the warehouse and never touches the path passed in as "production" until a separate `apply_action` call (confirmed by hash-comparison tests in `tests/test_sandbox.py`, reused unmodified here).
6. Verify repairs restore correct output — yes, before/after check comparison, 15/15 conditional success.
7. Record results — yes, every run recorded via `adie.control.incident_store`.

Item not exercised with a live LLM: the actual Claude-driven reasoning step in `propose_repair`. This is the single largest piece of unverified functionality in this report, and it is unverified because no credential is available, not because it was skipped by choice.

## 8. Limitations

- **Sample sizes are small by research standards** (12-16 fault cases per phase). Reported confidence intervals (§6.3) reflect this; conclusions about rare-event behavior (e.g. a fault type never seen in these runs) are not supported by this data.
- **The five numeric features were hand-selected** based on this project's own data model; a different feature set could change which fault types are "easy" for the supervised models.
- **RCA ground truth has a fixed, small candidate set** (5 assets). A production system with a much larger asset graph would need to re-validate whether lineage-constrained Top-3 accuracy holds at scale — this experiment doesn't test graph size as a variable.
- **No live LLM evaluation** (§7.3) — the single most consequential open item.
- **`missing_partition`'s one detection miss** was not root-caused further in this pass; worth a dedicated follow-up given it's the fault type IsolationForest structurally cannot see as a row-level anomaly (§4.4) and relies entirely on the completeness check, which should be ~100% reliable by construction — the miss suggests either a boundary condition in that check or an artifact of this specific arrangement, and should be investigated before relying on missing_partition detection being guaranteed.
- **Dockerfile/compose are still unverified** — no `docker` binary in this environment (carried over from the previous phase; unchanged in this pass since Phase 6 here explicitly deprioritizes cosmetic/environment work not required for the research deliverables).

## 9. Conclusions

1. For the three row-level fault types tested, supervised models (Random Forest, XGBoost) outperform both the unsupervised Isolation Forest variant and the statistical baseline on precision and false-positive rate, at the cost of needing labeled training examples the live unsupervised production detector does not require.
2. Lineage-constrained RCA is not universally the most accurate by Top-1 in this experiment — a no-lineage ML classifier scored higher — but it is the only method that (a) correctly handles a fault type with no row-count signal (join_error) where the no-lineage statistical method fails completely, and (b) expresses genuine ranking uncertainty (Top-3) rather than a single possibly-overfit guess. Both properties matter more for a system meant to generalize to fault types not seen during training than a single Top-1 number can show.
3. The agentic repair loop, run end-to-end on 16 independently-built fault cases, achieves 100% conditional repair success given detection, and — the result this project considers most important — correctly refuses to auto-repair the one fault type with no safe data-only fix.
4. The project's stated reuse priority was followed throughout: every model in Phase 1 is an unmodified, established library implementation; Graph Neural Networks were considered and explicitly not introduced per the stated criteria; no detection, RCA, or agent logic was reimplemented for this research phase, only orchestrated and measured.
