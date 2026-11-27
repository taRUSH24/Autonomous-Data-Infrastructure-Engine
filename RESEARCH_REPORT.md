# Research Report: Autonomous Data Infrastructure Engine

**Date:** 2026-11-25
**Experiment run:** `experiments/results/summary.json`, generated 2026-11-25T00:16:36 by `scripts/run_research_experiments.py`

> **Update 2026-11-27 — see [Part II](#part-ii--hybrid-explainable-anomaly-detection-2026-11-27).**
> Part I below is preserved unchanged as the record of the first experiment. Part II re-examined it
> and found evaluation issues that make Part I's perfect supervised scores optimistic
> (pseudo-replicated clean rows, full-severity-only faults, and a data sample with constant
> `trip_count` covering only ~00:00–06:00 of each day). Part II's numbers supersede Part I's
> anomaly-detection numbers; Part I's RCA and repair-loop results are not re-measured there.

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

---

# Part II — Hybrid Explainable Anomaly Detection (2026-11-27)

**Run:** `scripts/run_hybrid_experiments.py` → `experiments/results/hybrid/` (`summary.json`, `model_comparison.csv`, …).
Two independent runs produced identical metrics (max absolute difference 0.0). The operational artifact
(`models/hybrid_ensemble/`) is produced by `scripts/train_hybrid_ensemble.py` with the same code, data and
validation-only choices. Every number below is copied from those files.

## II.1 What was wrong with the Part I evaluation

| Issue | Effect | Fix in Part II |
|---|---|---|
| `fetch_*` took the **first N trips of each day ordered by pickup time** | every day had exactly N trips (`trip_count` constant by construction) and only ~00:00–06:00 trips were seen | uniform sample of the whole day (pickup-second = 7, ≈1/60 of all trips); `trip_count` now varies with real demand |
| each fault arrangement re-contributed all its **unfaulted** days | each real clean day appeared ~20× in train and test (pseudo-replication) | clean rows come from the clean arrangement exactly once; fault arrangements contribute only faulted days |
| all faults were **full-severity** (3× duplication, 100 % broken zone ids) | near-deterministic separability → F1 = 1.0 | graded severities (fraction of the day's rows affected) |
| 28 + 14 independent days | small, one season | 351 real days, chronological 3-way split |
| the offline IF was fit on train rows **including faults** | contaminated "normal" model | unsupervised models fit on clean train days only |
| ECOD scores a query batch *jointly* with its training set | test points shift each other's scores | rows scored one at a time (`adie/detection/hybrid.py:ecod_raw`, tested) |

## II.2 Data

Real NYC TLC 2023 yellow-taxi trips from Socrata (`scripts/fetch_hybrid_data.py`), 365 days, 636,644 sampled trips,
pushed through the **real** pipeline (raw load → fault injection → `dbt build` → `fct_trips_daily`) for every
arrangement (`adie/eval/hybrid_data.py`). The five existing features are unchanged.

| Split | Dates | Clean days | Injected-fault rows | Purpose |
|---|---|---|---|---|
| live demo | 2023-01-01 … 01-14 | 14 | — | Dagster demo warehouse; never used for fitting |
| train | 2023-01-15 … 06-30 | 165 | 294 | fit all models |
| validation | 2023-07-01 … 08-31 | 62 | 105 | calibration, thresholds, fusion weight, every design choice |
| test | 2023-09-01 … 12-31 | 116 | 142 | reported once |

Fault types (row-level, one fault per day, graded severity = fraction of the day's rows hit):
`duplicate_records` {0.1, 0.25, 0.5, 1.0}, `partial_partition` (new: truncated day) {0.1, 0.25, 0.5, 0.75},
`join_error` {0.02, 0.1, 0.5, 1.0}, `unit_change` (new: miles→km) {0.25, 0.5, 1.0}, `schema_change` (full only —
a *partial* null of `fare_amount` is invisible to `avg()`, see II.9). `missing_partition` produces no row, so it is
handled by the existing completeness rule (16 injected: 15 caught; the miss is 2023-08-31, the **last** day of the
validation range — the rule only checks gaps *inside* min..max and is blind at the boundary).

**Real incidents found in the published source.** Building this dataset surfaced genuine data-quality incidents
in Socrata itself: 2023-09-22/23/24 contain ~4,600–4,800 trips vs ~110,000 on neighbouring days (verified with
full-day `count(*)` queries; 09-21 is partial), and 2023-03-17, 05-11, 10-30, 10-31 each contain a physically
impossible trip (up to 18,469 miles in our sample; 131,376 miles in the full source on 10-31). These 8 days were
identified by a fixed rule on raw rows (`hybrid_data.natural_anomaly_days`: < 50 % of median volume and not a
pre-registered holiday, or any trip > 1,000 miles) — **not** by any detector — and held aside: excluded from clean
training/calibration and from false-positive rates, reported separately (II.6).

## II.3 Models and fusion (all library implementations)

| Branch | Model | Library | Fit on |
|---|---|---|---|
| unsupervised | IsolationForest | scikit-learn (same estimator as the live detector) | clean train days |
| unsupervised | ECOD | PyOD | clean train days |
| unsupervised | AutoEncoder [4-2-4] (evaluated, then **selected** — II.4) | PyOD + torch | clean train days |
| supervised | XGBoost (primary), RandomForest (baseline) | xgboost / scikit-learn, Part I hyper-parameters | clean + labelled fault train rows |

Custom code is limited to: **calibration** — each unsupervised raw score → its empirical-CDF position among clean
*validation* days, with an exponential tail above the largest clean score so it never saturates
(`TailCalibrator`); **combination** — average or max of calibrated scores; **fusion**:

* *weighted*: `s = w·p_sup + (1−w)·u`, `w` and threshold grid-searched on validation (selected `w` = 0.15);
* *gated*: alarm if `p_sup ≥ t_sup` (known-fault pattern) **or** `u ≥ t_unsup` (novelty). `t_unsup` is label-free:
  at most 2 % of clean validation days above it (0.968). `t_sup` then maximises validation F1 with total
  validation FPR ≤ 5 % (0.962). Exposed as one score `max(p/t_sup, u/t_unsup)`, alarm at ≥ 1.

Every threshold, for every model, maximises validation F1 subject to validation FPR ≤ 5 %. Test data is used for nothing but the numbers below.

## II.4 Validation-only design decisions

**E0 — unsupervised members** (`unsupervised_design_validation.csv`; validation PR-AUC):

| Members / combiner | PR-AUC | Recall @ FPR ≤ 5 % |
|---|---|---|
| IF + ECOD, average | 0.810 | 0.181 |
| IF + ECOD, max | 0.849 | 0.314 |
| IF + ECOD + AE, average | 0.861 | 0.238 |
| **IF + ECOD + AE, max** (selected) | **0.949** | **0.695** |
| individual: IF 0.813, ECOD 0.828, **AE 0.949** | | |

The AutoEncoder was added only because it clearly improved validation performance, as the brief required. The
reason is structural: **IF and ECOD scores saturate beyond the training range** (ECOD's per-feature tail score is
capped near `log(n+1)`; IF isolates "far out" and "just past the minimum" with the same splits), so a day whose
volume doubles looks no more anomalous than a day slightly outside the range on every feature. Reconstruction error
is unbounded. With 165 training days the AE is tiny (4-2-4) and its validation gain is large, so the data-volume concern did not block it.

**Model selection** (`model_selection_validation.csv`): score = ½·(validation F1, all types known) + ½·(validation
recall on the held-out type, averaged over leave-one-type-out folds), eligible if validation FPR ≤ 5 %.

| Candidate | Val F1 | Val FPR | Val unseen recall | Score |
|---|---|---|---|---|
| **hybrid_gated** (selected) | 0.832 | 0.048 | 0.680 | **0.756** |
| unsup_ensemble | 0.807 | 0.048 | 0.680 | 0.743 |
| autoencoder | 0.802 | 0.016 | 0.680 | 0.741 |
| isolation_forest_live (deployed baseline) | 0.774 | **0.145** (ineligible) | 0.690 | 0.732 |
| hybrid_weighted | 0.875 | 0.048 | 0.223 | 0.549 |
| random_forest | 0.881 | 0.048 | 0.077 | 0.479 |
| xgboost | 0.839 | 0.048 | 0.096 | 0.468 |

> **Disclosure.** The legacy statistical z-score baseline (|z| > 3 on any feature) was registered as
> research-only and so was not in this pool. Scored afterwards on the *same validation criterion* it gets
> **0.785** (F1 0.832, FPR 0.048, unseen recall 0.738) — it **would have been selected** over the hybrid.
> We noticed its strength on test data, so we did not re-design the ensemble around it (that would leak test
> information into the design); see II.8.

Localization (which feature a Signal names, the hint RCA consumes) was also chosen on validation faults:
SHAP-supervised 0.952, z-score 0.914, ECOD 0.876 → `shap_supervised`.

## II.5 Results on TEST (Sep–Dec 2023)

**E1 known faults + E3 clean days** (`model_comparison.csv`; 142 fault rows, 116 clean days; FPR 95 % Wilson CI):

| Model | Prec. | Recall | F1 | FPR [95 % CI] | ROC-AUC | PR-AUC | Prec. @ 5 % prevalence | ms / single day |
|---|---|---|---|---|---|---|---|---|
| IsolationForest — live per-batch (**current default before this change**) | 0.851 | 0.725 | 0.783 | 0.155 [0.100, 0.232] | – | – | 0.197 | 159.9 |
| IsolationForest — calibrated | 0.788 | 0.366 | 0.500 | 0.121 [0.073, 0.192] | 0.749 | 0.738 | 0.138 | 22.4 |
| ECOD | 0.846 | 0.310 | 0.454 | 0.069 [0.035, 0.130] | 0.804 | 0.790 | 0.191 | 22.4 |
| AutoEncoder | 0.921 | 0.739 | 0.820 | 0.078 [0.041, 0.141] | 0.912 | 0.936 | 0.334 | 22.7 |
| Unsup. ensemble (IF+ECOD+AE, max) | 0.908 | 0.761 | 0.828 | 0.095 [0.054, 0.162] | 0.911 | 0.935 | 0.297 | 22.9 |
| XGBoost | **0.922** | 0.838 | **0.878** | 0.086 [0.047, 0.151] | 0.943 | 0.955 | 0.338 | **2.6** |
| RandomForest | 0.910 | **0.852** | **0.880** | 0.103 [0.060, 0.172] | **0.945** | **0.957** | 0.302 | 18.1 |
| Hybrid — weighted | 0.896 | 0.845 | 0.870 | 0.121 [0.073, 0.192] | 0.929 | 0.952 | 0.269 | 50.7 |
| **Hybrid — gated (selected)** | 0.906 | 0.810 | 0.855 | 0.103 [0.060, 0.172] | 0.929 | 0.951 | 0.292 | 50.6 |
| z-score \|z\|>3 (legacy research baseline) | **0.924** | 0.768 | 0.838 | **0.078** [0.041, 0.141] | – | – | **0.342** | – |
| IsolationForest fit-once, contamination 0.1 (Part I variant) | 0.721 | 0.711 | 0.716 | 0.336 [0.257, 0.426] | – | – | 0.100 | – |

Precision on the test set is flattered by its ~55 % fault prevalence; the "@ 5 % prevalence" column re-weights
recall/FPR to one bad day in twenty. Latency is wall-clock on this machine; the hybrid's 50 ms is dominated by
ECOD's one-row-at-a-time scoring and is irrelevant for a daily batch.

**Recall by fault type (E1):**

| Model | duplicate | partial_partition | join_error | unit_change | schema_change |
|---|---|---|---|---|---|
| IF live | 0.567 | 0.448 | 0.966 | 0.704 | 0.963 |
| Unsup. ensemble | 0.733 | 0.621 | 1.000 | 0.444 | 1.000 |
| XGBoost | 0.767 | 0.655 | 1.000 | 0.778 | 1.000 |
| Hybrid gated | 0.767 | 0.621 | 1.000 | 0.667 | 1.000 |
| z-score | 0.733 | 0.621 | 1.000 | 0.481 | 1.000 |

Misses are concentrated at low severity (`figures/detection_by_severity.png`): a 10 % duplication or truncation
moves `trip_count` less than ordinary weekday/weekend variation, and without calendar features no model here can
separate the two. (Each severity point rests on 4–12 cases; treat the curve shape, not individual points.)

**E2 leave-one-fault-type-out — detection of the UNSEEN type** (`lofo_results.csv`; the held-out type is removed
from supervised training *and* from validation calibration):

| Held out → | duplicate | join_error | partial_partition | schema_change | unit_change | **pooled** [95 % CI] |
|---|---|---|---|---|---|---|
| XGBoost | 0.07 | 0.17 | 0.03 | 0.33 | 0.52 | 0.218 [0.158, 0.293] |
| RandomForest | 0.07 | 0.17 | 0.03 | 0.22 | 0.04 | 0.106 [0.065, 0.167] |
| Hybrid weighted | 0.13 | 0.34 | 0.03 | 0.63 | 0.59 | 0.338 [0.265, 0.419] |
| **Hybrid gated** | 0.73 | 1.00 | 0.62 | 1.00 | 0.33 | **0.739** [0.662, 0.805] |
| Unsup. ensemble | 0.73 | 1.00 | 0.62 | 1.00 | 0.33 | 0.739 |
| IF live | 0.57 | 0.97 | 0.45 | 0.96 | 0.70 | 0.725 [0.647, 0.792] |
| z-score | 0.73 | 1.00 | 0.62 | 1.00 | 0.48 | 0.768 [0.692, 0.829] |

**Known vs unseen** (`known_vs_unseen.csv`, `figures/known_vs_unseen.png`): XGBoost 0.838 → 0.218, RandomForest
0.852 → 0.106, hybrid gated **0.810 → 0.739**. The supervised models recognise what they were shown and little else;
the gated hybrid keeps most of its detection when a fault type is new, because the novelty gate fires on its own.
Its unseen detection is exactly the unsupervised branch's: the supervised branch adds nothing on new types, as expected.
The weighted fusion (`w` re-tuned per fold on known types only) learns to trust the supervised score and loses the
unseen types — this is why the gate, not the weighted average, was selected.

## II.6 Legitimate distribution changes and real incidents

**Legitimate events (should not alarm)** — 9 pre-registered test-period events (`legit_event_alarms.csv`):

| Day | Event | IF live | XGB | Hybrid gated | z-score |
|---|---|---|---|---|---|
| 09-04 | Labor Day | ✗ alarm | ✗ | ✗ | ✗ |
| 09-29 | Flash-flood emergency | ✗ | ✓ | ✗ | ✗ |
| 10-09 | Columbus Day | ✗ | ✓ | ✓ | ✓ |
| 11-05 | NYC Marathon | ✓ | ✓ | ✓ | ✓ |
| 11-23 | Thanksgiving | ✗ | ✗ | ✗ | ✗ |
| 11-24 | Black Friday | ✓ | ✗ | ✓ | ✓ |
| 12-24 | Christmas Eve | ✓ | ✗ | ✗ | ✗ |
| 12-25 | Christmas Day | ✗ | ✗ | ✗ | ✗ |
| 12-31 | New Year's Eve | ✓ | ✓ | ✓ | ✓ |

The hybrid false-alarms on 5 / 9 event days (FPR 0.556) against 0.065 on ordinary clean days; **most of its test FPR
comes from holidays/weather**. Every model shares this, because none of the five features knows the calendar.
Each such alarm would reach the agent as a repair proposal; the human-approval gate and sandbox verification are
what stop it from becoming an unnecessary repair (a re-ingest of a correct day resolves nothing, so the revised
sandbox check rejects it — II.7). The same failure appears in the live demo window: both detectors flag
2023-01-02 (observed New Year holiday, 1,065 sampled trips).

**Real source incidents held aside** (`natural_source_anomalies_test.csv`): the 09-21…09-24 volume gap was flagged
by **every** model (4/4). The impossible-trip days: hybrid gated 2/2, XGBoost 2/2, IF live 2/2, z-score 1/2,
ECOD 0/2. Hybrid gated 6/6 overall. These are real, unscripted upstream failures in the public dataset; the
hybrid's two branches together catch all of them (n = 6 — anecdotal, not a rate).

## II.7 Explainable AI (SHAP, LIME) — results and what they mean

All explanations are computed on **test** days with models fit on **train** only; LIME's sampling statistics come
from the training split only.

* **What is being explained.** Supervised SHAP/LIME explain the *fault probability* (TreeSHAP on XGBoost is in
  log-odds; on RandomForest in probability). IsolationForest TreeSHAP explains *path length*; ECOD's per-feature
  tail scores are exact additive contributions to its *anomaly score*; the calibrated ensemble score is explained
  model-agnostically (permutation SHAP with 50 clean train days as background; LIME in regression mode). The
  pipeline labels which one it shows (`primary_explains`).
* **Global importance (XGBoost, mean |SHAP|, log-odds)** — `global_top5_features.csv`: 1 `trip_count` 1.93,
  2 `avg_trip_distance_miles` 1.37, 3 `avg_fare_amount` 1.19, 4 `unmatched_zone_count` 0.87, 5 `avg_total_amount` 0.44.
  `trip_count` ranks first for every model (`shap_global_importance.csv`). With five features, "top five" is the
  complete ranking; `unmatched_zone_count` ranks low globally only because it is non-zero on few days, while being
  decisive when it is (`figures/shap_beeswarm_xgboost.png`).
* **Localization accuracy** (top-1 attributed feature = the feature the fault corrupts; `explanation_localization_test.csv`):
  SHAP-RandomForest 0.99, z-score 0.97, SHAP-XGBoost 0.96, LIME-XGBoost 0.95, SHAP-ensemble 0.93, ECOD 0.91,
  LIME-ensemble 0.66. LIME on the *ensemble* score is the weak one: the gated score is a max of two branches, and
  LIME's local linear surrogate over discretised bins fits that kink poorly (e.g. it scores 0.00 on partial_partition).
* **SHAP vs LIME (XGBoost, 115 detected test faults; `shap_vs_lime.csv`)**: top-1 feature agreement 0.92, top-3
  Jaccard 0.58, Spearman of |attributions| 0.39. They agree on *what drove the alarm*; they disagree on the ordering
  of the minor features, which carry little signal either way.
* **Representative anomalies** (one median-severity detected case per fault type): `figures/explanations/
  waterfall_xgboost_*.png`, `waterfall_ensemble_*.png`, `lime_xgboost_*.png`, `lime_ensemble_*.png`,
  `lime_representative_anomalies.json` (top-5 lists + LIME rules). Per-prediction top-5 for every flagged test day
  and seven attribution methods: `top5_features_per_prediction.csv`.

**What SHAP/LIME do *not* do.** They say which aggregate feature moved, not which pipeline asset caused it. Root
cause stays with the lineage-constrained RCA (`adie/rca/scorer.py`, unchanged). Feeding the hybrid's localized
metric into it (`rca_handoff_test.csv`): top-1 asset 0.61 (SHAP), 0.62 (ECOD), 0.60 (z-score); **top-2 1.00** for
all. The top-1 misses are schema_change and unit_change, whose value-level metrics the RCA scorer deliberately
credits to both `trips` and `stg_trips` (Part I §6.2) — an RCA design property, not an explanation error.

## II.8 Discussion — did the hybrid help?

* **Against the detector the pipeline actually ran** (per-batch IsolationForest): yes on every axis — F1 0.855 vs
  0.783, FPR 0.103 vs 0.155, unseen-type detection 0.739 vs 0.725, 6/6 vs 6/6 real incidents, and no forced
  10 %-of-every-batch alarm quota. It also detects the code-level `transformation_defect` (every day's fare doubled)
  on 14/14 demo days, which the per-batch IF structurally cannot (a uniform shift is invisible relative to its own batch).
* **Against XGBoost alone**: worse on known faults (recall 0.810 vs 0.838, F1 0.855 vs 0.878) and slightly higher
  FPR (0.103 vs 0.086); far better on unseen faults (0.739 vs 0.218). The known-fault cost is the price of the
  gate: the novelty gate uses part of the 5 % validation FPR budget, so `t_sup` must rise to 0.962.
* **Against the simple z-score baseline: not better.** Within overlapping confidence intervals the z-score matches
  or beats the hybrid on precision, FPR and unseen detection, losing only on known-fault recall (0.768 vs 0.810).
  The injected faults each corrupt one feature, and an unbounded per-feature deviation is exactly the right
  statistic for that. The tree/ECDF detectors saturate; the AutoEncoder is what rescues the unsupervised branch.
  An honest reading: **with five aggregate features, most of the hybrid's detection value comes from the
  AutoEncoder and XGBoost; its distinct contributions are (a) graded fault probability + novelty labelling
  (`supervised_known_fault` / `unsupervised_novel`), (b) SHAP/LIME explanations, and (c) known-fault recall.**
* **Uncertainty.** 116 clean test days give FPR intervals ~±5 pp wide; the top five models' differences are mostly
  inside them. Test FPRs (7–16 %) exceed the 5 % validation budget for every model: the test period is a later
  season with more holidays (and the validation period is summer).

Because the deployed baseline is clearly weaker, the hybrid is the configured default (`ADIE_DETECTOR=hybrid`, from
the artifact's validation-selected recommendation). The original IsolationForest remains one setting away and is
the automatic fallback. Whether to promote the z-score baseline into the ensemble is a decision to make **on
fresh data** (e.g. 2024) after pre-registering it — not on this test set.

## II.9 Limitations

1. **Five aggregate features, no calendar.** Holidays drive most false alarms; low-severity volume faults are
   indistinguishable from demand swings. Day-of-week/holiday features are the obvious next step (not added here
   because the brief fixed the five features).
2. **Injected faults are synthetic** (real data, simulated corruption). Each corrupts essentially one feature;
   multi-feature or gradual faults are not represented. A partial `schema_change` is undetectable by design
   (`avg()` ignores NULLs) and was therefore not injected.
3. **Leave-one-type-out with five types** is a coarse proxy for "unseen in production".
4. **Single year, single dataset, ~1/60 sample**; 365 independent days is better than 42 but still modest.
5. Validation serves both conformal calibration and threshold selection (all non-test, but slightly optimistic,
   which is part of why test FPR > 5 %).
6. Completeness rule cannot see a missing day at the edge of the range (1/16 missed).
7. Single-threaded wall-clock latency on one machine.

## II.10 Operational role of every model

| Model | In the operational pipeline | In research |
|---|---|---|
| Hybrid gated (IF + ECOD + AE ‖ XGBoost) | **Default detector** (`ADIE_DETECTOR=hybrid`); pretrained artifact, loaded not retrained | evaluated E1–E4 |
| XGBoost | supervised branch of the hybrid; its TreeSHAP is the primary explanation and localization | standalone comparison |
| RandomForest | stored in the artifact as the supervised comparison baseline; not used for decisions | standalone comparison |
| IsolationForest (calibrated), ECOD, AutoEncoder | unsupervised members (max of calibrated scores); ECOD attributions included in each report | standalone comparisons |
| IsolationForest per-batch (original) | **baseline + automatic fallback** (`ADIE_DETECTOR=isolation_forest`); its Dagster check still runs | evaluated as deployed |
| z-score \|z\|>3, IF fit-once | none | research baselines only |
| Calendar-completeness rule | always on (missing days have no row for any ML model) | II.2 |
