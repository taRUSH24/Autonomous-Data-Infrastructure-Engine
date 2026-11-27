"""
Detection + explanation step of the pipeline, as a Dagster asset downstream
of the dbt model it watches:

    raw_trips -> dbt (stg_* -> trips_enriched -> fct_trips_daily)
        -> anomaly_explanation_report   (this asset: configured detector + SHAP/LIME)
            -> investigate_and_repair_job (lineage RCA -> LangGraph -> sandbox verify)

The detector is chosen by ADIE_DETECTOR (adie/detection/selector.py). The
report lands in data/warehouse/explanations/ (JSON + SHAP waterfall PNGs)
and is summarised in the asset's Dagster metadata, so it renders in the
existing Dagster UI -- no separate dashboard.
"""

import json

import dagster as dg
from dagster import AssetExecutionContext

from dagster_project.resources import DUCKDB_PATH, EXPLANATIONS_DIR


def _md_table(signals) -> str:
    if not signals:
        return "_No anomalous days._"
    lines = ["| day | branch | score | p_fault | unsup | localized metric | top SHAP (fault prob.) | top LIME |",
             "|---|---|---|---|---|---|---|---|"]
    for s in signals:
        r, e = s.raw, s.raw.get("explanation", {})
        shap3 = ", ".join(f"{d['feature']} {d['contribution']:+.2f}" for d in e.get("supervised_shap_top5", [])[:3])
        lime3 = ", ".join(f"{d['feature']} {d['contribution']:+.2f}" for d in e.get("lime_top5", [])[:3])
        lines.append(f"| {r.get('pickup_date')} | {r.get('branch', '-')} | {r.get('ensemble_score', float('nan')):.3f} "
                     f"| {r.get('p_sup', float('nan')):.3f} | {r.get('u_unsup', float('nan')):.3f} | {s.metric} "
                     f"| {shap3 or '-'} | {lime3 or '-'} |")
    lines.append("")
    lines.append("_Attributions describe which aggregate features moved; root cause is decided by "
                 "lineage RCA in `investigate_and_repair_job`, not by SHAP/LIME._")
    return "\n".join(lines)


@dg.asset(
    group_name="detection",
    deps=[dg.AssetKey(["fct_trips_daily"])],
    description=(
        "Runs the configured anomaly detector (ADIE_DETECTOR: hybrid ensemble or original IsolationForest) "
        "over fct_trips_daily and writes per-day SHAP/LIME explanations for every flagged day."
    ),
)
def anomaly_explanation_report(context: AssetExecutionContext) -> dg.MaterializeResult:
    from adie.detection import selector

    signals, used = selector.detect_row_level(str(DUCKDB_PATH))
    if "fallback" in used:
        context.log.warning("hybrid artifact unavailable -- fell back to the original IsolationForest detector")
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)

    plots = []
    if used == "hybrid" and signals:
        import numpy as np

        from adie.detection import hybrid
        from adie.detection.isolation_forest import FEATURE_COLUMNS
        from adie.explain.attributions import save_waterfall

        model = hybrid.load(selector.artifact_path())
        for s in signals:
            x = np.array([s.raw[c] if s.raw[c] is not None else 0.0 for c in FEATURE_COLUMNS])
            day = s.raw["pickup_date"]
            plots.append(save_waterfall(model, x, EXPLANATIONS_DIR / f"waterfall_{day}.png",
                                        f"{day}: SHAP for XGBoost P(fault) (log-odds)"))

    report = {"detector": used, "signals": [s.to_dict() for s in signals], "waterfall_plots": plots}
    path = EXPLANATIONS_DIR / "latest.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    context.log.info(f"[{used}] {len(signals)} anomalous day(s); report -> {path}")
    return dg.MaterializeResult(metadata={
        "detector": used,
        "anomalous_day_count": len(signals),
        "anomalous_days": [s.raw.get("pickup_date") for s in signals],
        "explanations": dg.MetadataValue.md(_md_table(signals)),
        "report_path": dg.MetadataValue.path(str(path)),
    })
