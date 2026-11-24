"""
Root-cause plane: scores the upstream cone of a failing asset instead of
running causal discovery over all assets.

The scoring itself is adapted from the random-walk / closeness heuristic
used in Salesforce's PyRCA (salesforce/PyRCA, BSD-3): walk the dependency
graph backwards from the failing node, and rank ancestors by a combination
of (a) graph proximity and (b) how many independent signals implicate them.
PyRCA's own implementation assumes it has to *discover* the graph; we don't
need that step because dbt's manifest already gives us ground truth, so
only the ranking heuristic is reused here, not the whole library.
"""
from __future__ import annotations

from collections import defaultdict

from adie.contracts import Hypothesis, Signal
from adie.lineage.graph import LineageGraph

# Mechanism labels are informative hints for the agent plane, not causal claims.
MECHANISM_BY_NODE_HINT = {
    "trips": "upstream_source_quality",
    "stg_trips": "transformation_defect",
    "stg_zones": "dimension_defect",
    "trips_enriched": "join_defect",
    "fct_trips_daily": "aggregation_defect",
}

# Which mechanism(s) a given anomalous *metric* implicates, independent of
# which asset tripped the check. These follow from what each column means,
# not from knowing which fault produced it -- the per-metric localization
# step PyRCA calls epsilon-diagnosis, reused here as a lookup table instead
# of its full implementation since our metric set is small and fixed.
#
# unmatched_zone_count and a vanished day are structurally unambiguous in
# this DAG (only the join step, or only raw deletion, can produce them).
# A value-level shift (avg_fare_amount etc.) genuinely can't be told apart
# between "the source already had bad values" and "the cast/transform step
# corrupted them" from the aggregate alone -- so it gets credited to both
# candidate mechanisms rather than forced to a false-precise single answer.
MECHANISM_BY_METRIC = {
    "missing_partition": ["upstream_source_quality"],
    "trip_count": ["upstream_source_quality"],
    "unmatched_zone_count": ["join_defect", "dimension_defect"],
    "avg_fare_amount": ["upstream_source_quality", "transformation_defect"],
    "avg_total_amount": ["upstream_source_quality", "transformation_defect"],
    "avg_trip_distance_miles": ["upstream_source_quality", "transformation_defect"],
}

MECHANISM_MATCH_BONUS = 2.0


def _mechanism_for(asset_name: str) -> str:
    # Exact match only: dbt model names are a small fixed set here, and a
    # substring check ("trips" in "trips_enriched") would misclassify every
    # name that happens to contain another node's name as a prefix.
    return MECHANISM_BY_NODE_HINT.get(asset_name, "unknown")


def rank_hypotheses(
    lineage: LineageGraph,
    signals: list[Signal],
    max_hops: int = 6,
) -> list[Hypothesis]:
    """
    For each Signal, constrain the search to that asset's *ancestors* (the
    symptom node itself is excluded -- "fct_trips_daily is anomalous" is not
    a root-cause hypothesis about fct_trips_daily, it's the reason we're
    looking), then score every candidate ancestor by:

        score = sum_over_signals( proximity_weight + mechanism_bonus )

    proximity_weight = 1 / (1 + hop_distance) is the graph-proximity term
    from PyRCA's random-walk scorer, applied to a known graph instead of a
    discovered one. mechanism_bonus adds extra weight when the candidate's
    mechanism matches what the signal's specific metric implicates (e.g. a
    signal driven by unmatched_zone_count favors the join step over the
    source), which is PyRCA's metric-localization idea applied via the
    MECHANISM_BY_METRIC table above instead of a learned model.
    """
    votes: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    supporting: dict[str, set[str]] = defaultdict(set)
    # Which specific days gave a candidate its mechanism-match bonus -- NOT
    # just "which check types touched this candidate" (check_name alone
    # can't disambiguate which day, since every day's isolation-forest
    # signal shares the same check_name).
    implicating_days: dict[str, set[str]] = defaultdict(set)

    for sig in signals:
        cone = lineage.upstream_cone(sig.asset_key)[:max_hops]
        implicated_mechanisms = MECHANISM_BY_METRIC.get(sig.metric, [])
        sig_day = sig.raw.get("pickup_date")

        for uid in cone:
            name = lineage.display_name(uid)
            dist = lineage.distance(name, sig.asset_key) or 0
            weight = 1.0 / (1.0 + dist)
            if _mechanism_for(name) in implicated_mechanisms:
                weight += MECHANISM_MATCH_BONUS
                if sig_day:
                    implicating_days[uid].add(sig_day)
            votes[uid] += weight
            evidence[uid].append(
                f"{sig.check_name} on {sig.asset_key} (hop={dist}, metric={sig.metric}, severity={sig.severity.value})"
            )
            supporting[uid].add(sig.check_name)

    hypotheses = []
    for uid, score in votes.items():
        name = lineage.display_name(uid)
        hypotheses.append(
            Hypothesis(
                asset_key=name,
                mechanism=_mechanism_for(name),
                score=round(score, 4),
                evidence=evidence[uid],
                distance_from_signal=min(
                    (lineage.distance(name, s.asset_key) or 0) for s in signals
                ) if signals else 0,
                supporting_signals=sorted(supporting[uid]),
                implicating_days=sorted(implicating_days[uid]),
            )
        )

    hypotheses.sort(key=lambda h: h.score, reverse=True)
    return hypotheses
