#!/usr/bin/env python3
"""
Hybrid-ensemble research runner (Phases 2-4, 6). Builds (or loads the cached)
labelled real-data feature set from adie.eval.hybrid_data, then:

  E0  unsupervised design choices on VALIDATION only (AutoEncoder in/out,
      average vs max combiner)
  E1  known faults: supervised branch trained on all 5 row-level fault types
  E2  leave-one-fault-type-out: each type removed from supervised training AND
      from validation calibration, then measured on TEST as an unseen type
  E3  clean test days (false-positive rate)
  E4  legitimate distribution changes (holidays / weather events in TEST)
  plus: latency, explanation localization accuracy, RCA hand-off accuracy,
  SHAP (TreeExplainer + permutation) and LIME explanations, SHAP-vs-LIME
  agreement, figures, and a validation-only model selection that
  scripts/train_hybrid_ensemble.py uses for the operational artifact.

Outputs -> experiments/results/hybrid/ (CSV, JSON, figures/*.png).
Run: .venv/bin/python scripts/run_hybrid_experiments.py [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adie.contracts import CheckKind, Severity, Signal  # noqa: E402
from adie.detection.hybrid import to_matrix  # noqa: E402
from adie.detection.isolation_forest import FEATURE_COLUMNS, score_daily_metrics  # noqa: E402
from adie.eval import hybrid_data, models as legacy_models  # noqa: E402
from adie.eval.hybrid_experiment import (  # noqa: E402
    candidate_predictions, fit_ensemble, latency_per_row, metrics,
)
from adie.eval.metrics import wilson_interval  # noqa: E402
from adie.explain import attributions as A  # noqa: E402
from adie.lineage.graph import LineageGraph  # noqa: E402
from adie.rca.scorer import rank_hypotheses  # noqa: E402

warnings.filterwarnings("ignore")
OUT = REPO_ROOT / "experiments" / "results" / "hybrid"
FIG = OUT / "figures"
TYPES = hybrid_data.ROW_FAULT_TYPES
MANIFEST = REPO_ROOT / "dbt_project" / "target" / "manifest.json"

# fixed entity -> colour (reference categorical palette, fixed order, never cycled)
COLORS = {
    "isolation_forest_live": "#2a78d6", "isolation_forest": "#eb6834", "ecod": "#1baf7a",
    "unsup_ensemble": "#eda100", "random_forest": "#e87ba4", "xgboost": "#008300",
    "hybrid_weighted": "#6250d6", "hybrid_gated": "#e34948",
}
LABELS = {
    "isolation_forest_live": "IsolationForest (live, per-batch)", "isolation_forest": "IsolationForest (calibrated)",
    "ecod": "ECOD", "unsup_ensemble": "Unsup. ensemble", "random_forest": "RandomForest",
    "xgboost": "XGBoost", "hybrid_weighted": "Hybrid: weighted", "hybrid_gated": "Hybrid: gated",
    "autoencoder": "AutoEncoder", "statistical_zscore": "z-score |z|>3 (legacy)",
    "isolation_forest_fit_once_legacy": "IsolationForest fit-once, contamination=0.1 (legacy)",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def style(ax, title=None, ylabel=None):
    ax.set_facecolor("#fcfcfb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=8)


# ---------------------------------------------------------------------------
# baselines that are not part of the ensemble object
# ---------------------------------------------------------------------------

def live_isolation_forest(eval_df: pd.DataFrame, clean_pool: pd.DataFrame, window: int = 14) -> np.ndarray:
    """The detector the pipeline actually ran before this change:
    adie.detection.isolation_forest.score_daily_metrics, refit on every
    batch. Each evaluated day is scored inside a 14-day batch made of the 13
    nearest *clean* days of the same split -- one fault per batch, like the
    live pipeline would see."""
    pool = clean_pool.copy()
    pool["_d"] = pd.to_datetime(pool.pickup_date)
    preds = np.zeros(len(eval_df), dtype=bool)
    for i, (_, r) in enumerate(eval_df.iterrows()):
        d = pd.Timestamp(r.pickup_date)
        others = pool[pool._d != d].assign(_dist=lambda x: (x._d - d).abs()).nsmallest(window - 1, "_dist")
        batch = pd.concat([others[FEATURE_COLUMNS], r[FEATURE_COLUMNS].to_frame().T], ignore_index=True).astype(float)
        preds[i] = bool(score_daily_metrics(batch)["is_anomaly"].iloc[-1])
    return preds


def legacy_baselines(train_clean, eval_df):
    out = {}
    m = legacy_models.fit_isolation_forest(train_clean)
    out["isolation_forest_fit_once_legacy"] = (None, m.predict(eval_df)[0].astype(bool))
    z = legacy_models.fit_statistical_baseline(train_clean)
    out["statistical_zscore"] = (None, z.predict(eval_df)[0].astype(bool))
    return out


# ---------------------------------------------------------------------------
# E0: unsupervised design choices on validation
# ---------------------------------------------------------------------------

def e0_unsupervised_design(train, val, ae_available):
    log("E0: unsupervised design choices (validation only)")
    rows = []
    member_sets = [["isolation_forest", "ecod"]] + ([["isolation_forest", "ecod", "autoencoder"]] if ae_available else [])
    best = None
    for members in member_sets:
        for combiner in ("average", "max"):
            m = fit_ensemble(train, val, TYPES, members, combiner)
            Xv, yv = to_matrix(val), val.is_fault.to_numpy()
            ms = m.member_scores(Xv)
            u = m.unsup_score(Xv, ms)
            r = {"members": "+".join(members), "combiner": combiner, **{
                f"val_{k}": v for k, v in metrics(yv, u >= m.thresholds["unsup_ensemble"], u).items()
                if k in ("roc_auc", "pr_auc", "recall", "fpr", "f1")}}
            for n, s in ms.items():
                r[f"val_pr_auc_{n}"] = metrics(yv, s >= 0.5, s)["pr_auc"]
            rows.append(r)
            key = (round(r["val_pr_auc"], 3), -len(members), combiner == "average")  # ties -> simpler
            if best is None or key > best[0]:
                best = (key, members, combiner)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "unsupervised_design_validation.csv", index=False)
    log(df.to_string())
    return best[1], best[2], df


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def predict_all(m, df, train_clean, live_pred):
    preds = candidate_predictions(m, df)
    preds["isolation_forest_live"] = (None, live_pred)
    preds.update(legacy_baselines(train_clean, df))
    return preds


def long_predictions(df, preds, experiment, fold):
    rows = []
    for name, (score, pred) in preds.items():
        rows.append(pd.DataFrame({
            "experiment": experiment, "fold": fold, "model": name,
            "pickup_date": df.pickup_date.values, "arrangement": df.arrangement.values,
            "fault_type": df.fault_type.values, "severity": df.severity.values,
            "legit_event": df.legit_event.values, "y_true": df.is_fault.values,
            "score": score if score is not None else np.nan, "y_pred": pred.astype(int),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# explanations
# ---------------------------------------------------------------------------

def explanations(m, test, train, pred_df):
    log("SHAP / LIME explanations on TEST (models fit on TRAIN only)")
    efig = FIG / "explanations"
    efig.mkdir(parents=True, exist_ok=True)
    import shap

    Xt = to_matrix(test)
    out = {}

    # --- global SHAP for both supervised models (fault probability) --------
    glob_rows = []
    for name in ("xgboost", "random_forest"):
        exp = A.tree_shap(m.models[name], Xt)
        exp.data = Xt
        imp = np.abs(exp.values).mean(axis=0)
        for f, v in zip(FEATURE_COLUMNS, imp):
            glob_rows.append({"model": name, "explains": "fault probability", "method": "TreeSHAP",
                              "output_space": "log-odds" if name == "xgboost" else "probability",
                              "feature": f, "mean_abs_shap": v})
        plt.figure(); shap.plots.beeswarm(exp, show=False, color_bar=True)
        plt.title(f"SHAP beeswarm - {LABELS[name]} (TEST days)", fontsize=10, loc="left")
        plt.tight_layout(); plt.savefig(FIG / f"shap_beeswarm_{name}.png", dpi=150); plt.close("all")
        plt.figure(); shap.plots.bar(exp, show=False)
        plt.title(f"Mean |SHAP| - {LABELS[name]}", fontsize=10, loc="left")
        plt.tight_layout(); plt.savefig(FIG / f"shap_bar_{name}.png", dpi=150); plt.close("all")
        out[f"shap_{name}"] = exp

    # --- unsupervised: IsolationForest TreeSHAP, ECOD intrinsic ------------
    exp_if = A.tree_shap(m.models["isolation_forest"], Xt)
    exp_if.data = Xt
    for f, v in zip(FEATURE_COLUMNS, np.abs(exp_if.values).mean(axis=0)):
        glob_rows.append({"model": "isolation_forest", "explains": "anomaly score (path length)",
                          "method": "TreeSHAP", "output_space": "path length (higher = more normal)",
                          "feature": f, "mean_abs_shap": v})
    plt.figure(); shap.plots.beeswarm(exp_if, show=False)
    plt.title("SHAP beeswarm - IsolationForest path length (lower = more anomalous)", fontsize=10, loc="left")
    plt.tight_layout(); plt.savefig(FIG / "shap_beeswarm_isolation_forest.png", dpi=150); plt.close("all")
    ecod_c = A.ecod_contributions(m.models["ecod"], Xt)
    for f, v in zip(FEATURE_COLUMNS, np.abs(ecod_c).mean(axis=0)):
        glob_rows.append({"model": "ecod", "explains": "anomaly score", "method": "ECOD per-feature tail score (exact)",
                          "output_space": "-log tail probability", "feature": f, "mean_abs_shap": v})

    # --- ensemble score: permutation SHAP on flagged test days --------------
    sel = pred_df[(pred_df.model == m.strategy) & (pred_df.y_pred == 1)]
    flagged = test.merge(sel[["pickup_date", "arrangement"]], on=["pickup_date", "arrangement"])
    Xf = to_matrix(flagged)
    bg = m.background[:50]
    exp_ens = A.score_shap(m.score_fn(), Xf, bg)
    exp_uns = A.score_shap(m.score_fn("unsup_ensemble"), Xf, bg)
    for name, e, what in [(m.strategy, exp_ens, "ensemble score"), ("unsup_ensemble", exp_uns, "calibrated anomaly score")]:
        for f, v in zip(FEATURE_COLUMNS, np.abs(e.values).mean(axis=0)):
            glob_rows.append({"model": name, "explains": what, "method": "Permutation SHAP (flagged days)",
                              "output_space": "score", "feature": f, "mean_abs_shap": v})
    glob = pd.DataFrame(glob_rows)
    glob["rank"] = glob.groupby("model")["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    glob.sort_values(["model", "rank"]).to_csv(OUT / "shap_global_importance.csv", index=False)
    top5 = glob[glob.model.isin(["xgboost"])].sort_values("rank")[["rank", "feature", "mean_abs_shap"]]
    top5.to_csv(OUT / "global_top5_features.csv", index=False)

    # --- per-prediction top-5 + LIME + localisation -------------------------
    lime_cls = A.lime_explainer(to_matrix(train))                       # TRAIN statistics only
    lime_reg = A.lime_explainer(to_matrix(train), mode="regression")
    xgb_flag = A.tree_shap(m.models["xgboost"], Xf).values
    rf_flag = A.tree_shap(m.models["random_forest"], Xf).values
    ecod_flag = A.ecod_contributions(m.models["ecod"], Xf)
    z_flag = m.zscores(Xf)
    per_rows, cmp_rows, loc_rows = [], [], []
    from scipy.stats import spearmanr

    for i, (_, r) in enumerate(flagged.iterrows()):
        x = Xf[i]
        lime_w, lime_rules, _ = A.lime_explain(lime_cls, x, m.models["xgboost"].predict_proba)
        lime_e, _, _ = A.lime_explain(lime_reg, x, m.score_fn())
        methods = {"shap_xgboost": xgb_flag[i], "shap_random_forest": rf_flag[i], "ecod": ecod_flag[i],
                   f"shap_{m.strategy}": exp_ens.values[i], "lime_xgboost": lime_w, f"lime_{m.strategy}": lime_e,
                   "zscore": np.abs(z_flag[i])}
        for meth, vals in methods.items():
            top = A._top(vals, x, positive_first=meth.startswith(("shap_x", "shap_r", "lime_x")))
            per_rows.append({"pickup_date": r.pickup_date, "arrangement": r.arrangement, "fault_type": r.fault_type,
                             "method": meth, **{f"top{k+1}": f"{d['feature']} ({d['contribution']:+.4f})"
                                                for k, d in enumerate(top)}})
            if r.is_fault:
                loc_rows.append({"fault_type": r.fault_type, "method": meth,
                                 "hit": int(top[0]["feature"] == r.expected_feature)})
        if r.is_fault:
            a, b = xgb_flag[i], lime_w
            ta, tb = set(np.argsort(-np.abs(a))[:3]), set(np.argsort(-np.abs(b))[:3])
            cmp_rows.append({
                "pickup_date": r.pickup_date, "fault_type": r.fault_type,
                "shap_top1": FEATURE_COLUMNS[int(np.argmax(np.where(a > 0, a, -np.inf)))] if (a > 0).any() else FEATURE_COLUMNS[int(np.argmax(np.abs(a)))],
                "lime_top1": FEATURE_COLUMNS[int(np.argmax(np.where(b > 0, b, -np.inf)))] if (b > 0).any() else FEATURE_COLUMNS[int(np.argmax(np.abs(b)))],
                "top3_jaccard": len(ta & tb) / len(ta | tb),
                "spearman_abs": spearmanr(np.abs(a), np.abs(b)).statistic,
                "lime_rules": " | ".join(f"{rule} ({w:+.3f})" for rule, w in lime_rules),
            })
    pd.DataFrame(per_rows).to_csv(OUT / "top5_features_per_prediction.csv", index=False)
    cmp = pd.DataFrame(cmp_rows)
    cmp["top1_agree"] = (cmp.shap_top1 == cmp.lime_top1).astype(int)
    cmp.to_csv(OUT / "shap_vs_lime.csv", index=False)
    loc = pd.DataFrame(loc_rows)
    loc_summary = loc.groupby(["method", "fault_type"]).hit.mean().unstack()
    loc_summary["overall"] = loc.groupby("method").hit.mean()
    loc_summary.to_csv(OUT / "explanation_localization_test.csv")

    # --- representative anomalies: waterfall (SHAP) + LIME bar per fault type
    reps = []
    for ftype in TYPES:
        cand = flagged[(flagged.fault_type == ftype)]
        if cand.empty:
            continue
        sev = cand.severity.median()
        idx = (cand.severity - sev).abs().idxmin()
        reps.append((ftype, flagged.index.get_loc(idx)))
    rep_json = []
    for ftype, i in reps:
        r, x = flagged.iloc[i], Xf[i]
        e = out["shap_xgboost"]
        ex = shap.Explanation(values=xgb_flag[i], base_values=float(np.ravel(e.base_values)[0]), data=x,
                              feature_names=FEATURE_COLUMNS)
        plt.figure(); shap.plots.waterfall(ex, show=False)
        plt.title(f"{ftype} on {r.pickup_date} (severity {r.severity:g}) - XGBoost log-odds", fontsize=9, loc="left")
        plt.tight_layout(); plt.savefig(efig / f"waterfall_xgboost_{ftype}.png", dpi=150); plt.close("all")
        ee = shap.Explanation(values=exp_ens.values[i], base_values=float(np.ravel(exp_ens.base_values)[i]), data=x,
                              feature_names=FEATURE_COLUMNS)
        plt.figure(); shap.plots.waterfall(ee, show=False)
        plt.title(f"{ftype} on {r.pickup_date} - {LABELS.get(m.strategy, m.strategy)} score", fontsize=9, loc="left")
        plt.tight_layout(); plt.savefig(efig / f"waterfall_ensemble_{ftype}.png", dpi=150); plt.close("all")
        w, rules, lexp = A.lime_explain(lime_cls, x, m.models["xgboost"].predict_proba)
        fig = lexp.as_pyplot_figure(label=1)
        fig.axes[0].set_title(""); fig.suptitle(f"LIME - XGBoost P(fault), {ftype} on {r.pickup_date}", fontsize=9, x=0.02, ha="left")
        fig.tight_layout(); fig.savefig(efig / f"lime_xgboost_{ftype}.png", dpi=150); plt.close("all")
        we, rules_e, lexp_e = A.lime_explain(lime_reg, x, m.score_fn())
        fig = lexp_e.as_pyplot_figure()
        fig.axes[0].set_title(""); fig.suptitle(f"LIME - ensemble score, {ftype} on {r.pickup_date}", fontsize=9, x=0.02, ha="left")
        fig.tight_layout(); fig.savefig(efig / f"lime_ensemble_{ftype}.png", dpi=150); plt.close("all")
        rep_json.append({
            "fault_type": ftype, "pickup_date": r.pickup_date, "severity": float(r.severity),
            "features": {f: float(v) for f, v in zip(FEATURE_COLUMNS, x)},
            "p_fault_xgboost": float(m.sup_proba(x[None])[0]), "ensemble_score": float(m.score_fn()(x[None])[0]),
            "shap_xgboost_top5": A._top(xgb_flag[i], x, positive_first=True),
            "shap_ensemble_top5": A._top(exp_ens.values[i], x),
            "ecod_top5": A._top(ecod_flag[i], x),
            "lime_xgboost_top5": A._top(w, x, positive_first=True), "lime_xgboost_rules": rules,
            "lime_ensemble_top5": A._top(we, x), "lime_ensemble_rules": rules_e,
        })
    (OUT / "lime_representative_anomalies.json").write_text(json.dumps(rep_json, indent=2))

    # SHAP vs LIME agreement figure
    agg = cmp.groupby("fault_type").agg(top1_agree=("top1_agree", "mean"), top3_jaccard=("top3_jaccard", "mean"))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = np.arange(len(agg))
    ax.bar(xs - 0.18, agg.top1_agree, 0.34, color="#2a78d6", label="top-1 feature agrees")
    ax.bar(xs + 0.18, agg.top3_jaccard, 0.34, color="#eb6834", label="top-3 Jaccard overlap")
    ax.set_xticks(xs, agg.index, fontsize=8); ax.set_ylim(0, 1.05)
    style(ax, "SHAP (TreeExplainer) vs LIME on XGBoost, detected TEST faults", "agreement")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "shap_vs_lime_agreement.png", dpi=150); plt.close(fig)

    return {"global": glob, "localization": loc_summary, "shap_vs_lime": cmp, "n_flagged_explained": int(len(flagged)),
            "representatives": [r["fault_type"] for r in rep_json]}


# ---------------------------------------------------------------------------
# RCA hand-off: does the localized metric lead lineage RCA to the right asset?
# ---------------------------------------------------------------------------

def rca_handoff(m, test, pred_df, localization_methods):
    if not MANIFEST.exists():
        return None
    lg = LineageGraph.from_manifest(str(MANIFEST))
    det = pred_df[(pred_df.model == m.strategy) & (pred_df.y_pred == 1) & (pred_df.y_true == 1)]
    rows = []
    sub = test.merge(det[["pickup_date", "arrangement"]], on=["pickup_date", "arrangement"])
    X = to_matrix(sub)
    shap_x = A.tree_shap(m.models["xgboost"], X).values
    ecod_c = A.ecod_contributions(m.models["ecod"], X)
    z = np.abs(m.zscores(X))
    for i, (_, r) in enumerate(sub.iterrows()):
        cand = {"zscore": z[i], "ecod": ecod_c[i],
                "shap_supervised": np.where(shap_x[i] > 0, shap_x[i], 0) if (shap_x[i] > 0).any() else np.abs(shap_x[i])}
        for meth in localization_methods:
            metric = FEATURE_COLUMNS[int(np.argmax(cand[meth]))]
            sig = Signal(asset_key="fct_trips_daily", check_name="hybrid_ensemble_daily_metrics",
                         kind=CheckKind.ANOMALY, severity=Severity.CRITICAL, metric=metric, observed=0.0,
                         expected=0.0, message="", raw={"pickup_date": r.pickup_date})
            hyps = rank_hypotheses(lg, [sig])
            ranked = [h.asset_key for h in hyps]
            rows.append({"method": meth, "fault_type": r.fault_type, "ground_truth_asset": r.ground_truth_asset,
                         "top1_hit": int(bool(ranked) and ranked[0] == r.ground_truth_asset),
                         "top2_hit": int(r.ground_truth_asset in ranked[:2])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "rca_handoff_test.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

CHART_MODELS = list(COLORS)


def fig_comparison(summary_e1: pd.DataFrame, path):
    metrics_ = [("f1", "F1"), ("recall", "Recall (known faults)"), ("fpr", "False-positive rate (clean days)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (k, title) in zip(axes, metrics_):
        d = summary_e1.set_index("model").reindex(CHART_MODELS)
        ax.bar(range(len(d)), d[k], color=[COLORS[mn] for mn in d.index], width=0.7, edgecolor="#fcfcfb", linewidth=2)
        for j, v in enumerate(d[k]):
            ax.text(j, v + 0.01 * d[k].max(), f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_ylim(0, d[k].max() * 1.18)
        ax.set_xticks(range(len(d)), [LABELS[mn] for mn in d.index], rotation=55, ha="right", fontsize=7)
        style(ax, title)
    fig.suptitle("E1 known faults - TEST (Sep-Dec 2023), thresholds from validation", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_lofo(lofo: pd.DataFrame, path):
    d = lofo.pivot(index="held_out", columns="model", values="unseen_recall").reindex(columns=CHART_MODELS)
    fig, ax = plt.subplots(figsize=(11, 4))
    n = len(CHART_MODELS); w = 0.8 / n
    for j, mn in enumerate(CHART_MODELS):
        ax.bar(np.arange(len(d)) + (j - n / 2 + 0.5) * w, d[mn], w * 0.9, color=COLORS[mn], label=LABELS[mn])
    ax.set_xticks(range(len(d)), d.index, fontsize=8); ax.set_ylim(0, 1.05)
    style(ax, "E2 leave-one-fault-type-out: detection rate of the UNSEEN fault type (TEST)", "detection rate")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_known_vs_unseen(kv: pd.DataFrame, path):
    d = kv.set_index("model").reindex(CHART_MODELS)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = np.arange(len(d))
    ax.bar(xs - 0.2, d.known_detection_rate, 0.38, color="#2a78d6", label="known fault types (E1)")
    ax.bar(xs + 0.2, d.unseen_detection_rate, 0.38, color="#eb6834", label="unseen fault types (E2, pooled)")
    ax.set_xticks(xs, [LABELS[mn] for mn in d.index], rotation=40, ha="right", fontsize=7); ax.set_ylim(0, 1.05)
    style(ax, "Known vs unseen fault detection rate (TEST)", "detection rate")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_severity(pred_e1: pd.DataFrame, path):
    fig, axes = plt.subplots(1, len(TYPES), figsize=(14, 3.2), sharey=True)
    for ax, ftype in zip(axes, TYPES):
        sub = pred_e1[(pred_e1.fault_type == ftype)]
        for mn in ["isolation_forest_live", "unsup_ensemble", "xgboost", "hybrid_gated"]:
            s = sub[sub.model == mn].groupby("severity").y_pred.mean()
            ax.plot(s.index, s.values, marker="o", markersize=4, linewidth=2, color=COLORS[mn], label=LABELS[mn])
        style(ax, ftype); ax.set_xlabel("fraction of day's rows affected", fontsize=7, color=INK2)
        ax.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("detection rate", fontsize=8, color=INK2)
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fig_scores(test, m, path):
    from sklearn.metrics import precision_recall_curve

    X, y = to_matrix(test), test.is_fault.to_numpy()
    c = m.candidate_scores(X)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mn in [k for k in CHART_MODELS if k in c]:
        p, r, _ = precision_recall_curve(y, c[mn])
        ax.plot(r, p, linewidth=2, color=COLORS[mn], label=LABELS[mn])
    style(ax, "Precision-recall, E1 TEST", "precision"); ax.set_xlabel("recall", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rebuild the dbt-derived feature dataset")
    ap.add_argument("--no-autoencoder", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    feats, missing = hybrid_data.build_or_load(rebuild=args.rebuild)
    feats.to_csv(OUT / "features.csv", index=False)
    missing.to_csv(OUT / "missing_partition_completeness.csv", index=False)
    # real source incidents on otherwise-unfaulted days: not clean, not injected -> held aside
    natural_mask = (feats.is_fault == 0) & (feats.natural_anomaly != "")
    natural = feats[natural_mask].reset_index(drop=True)
    feats = feats[~natural_mask]
    train, val, test = (feats[feats.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    log(f"held aside {len(natural)} natural source-anomaly days: "
        f"{natural[['split', 'pickup_date', 'natural_anomaly']].values.tolist()}")
    train_clean, val_clean, test_clean = (d[d.is_fault == 0] for d in (train, val, test))
    log(f"rows: train={len(train)} ({train.is_fault.sum()} faults), val={len(val)} ({val.is_fault.sum()}), "
        f"test={len(test)} ({test.is_fault.sum()}); clean days {len(train_clean)}/{len(val_clean)}/{len(test_clean)}")

    try:
        import torch  # noqa: F401
        ae_available = not args.no_autoencoder
    except ImportError:
        ae_available = False
    members, combiner, e0 = e0_unsupervised_design(train, val, ae_available)
    log(f"E0 decision: members={members}, combiner={combiner}")
    short = {"isolation_forest": "IF", "ecod": "ECOD", "autoencoder": "AE"}
    LABELS["unsup_ensemble"] = f"Unsup. ensemble ({'+'.join(short[m] for m in members)}, {combiner})"

    log("live IsolationForest baseline (per-batch refit) on TEST and VAL")
    live_test = live_isolation_forest(test, test_clean)
    live_val = live_isolation_forest(val, val_clean)

    # ---------------- E1 known faults ---------------------------------
    log("E1: all fault types known to the supervised branch")
    m1 = fit_ensemble(train, val, TYPES, members, combiner)
    val_preds = predict_all(m1, val, train_clean, live_val)
    val_e1 = {k: metrics(val.is_fault, p, s) for k, (s, p) in val_preds.items()}
    preds = predict_all(m1, test, train_clean, live_test)
    pred_rows = [long_predictions(test, preds, "E1_known", "all")]
    e1 = []
    for name, (score, pred) in preds.items():
        r = {"model": name, **metrics(test.is_fault, pred, score)}
        clean_mask = (test.is_fault == 0).to_numpy()
        ev = clean_mask & (test.legit_event != "").to_numpy()
        reg = clean_mask & (test.legit_event == "").to_numpy()
        r["fpr_legit_event_days"] = float(pred[ev].mean()) if ev.any() else np.nan
        r["fpr_regular_clean_days"] = float(pred[reg].mean()) if reg.any() else np.nan
        _, lo, hi = wilson_interval(int(pred[clean_mask].sum()), int(clean_mask.sum()))
        r["fpr_ci95"] = f"[{lo:.3f}, {hi:.3f}]"
        for ftype in TYPES:
            mk = (test.fault_type == ftype).to_numpy()
            r[f"recall_{ftype}"] = float(pred[mk].mean())
        r["val_f1"] = val_e1[name]["f1"]; r["val_fpr"] = val_e1[name]["fpr"]
        e1.append(r)
    e1 = pd.DataFrame(e1)

    # ---------------- E2 leave-one-fault-type-out ----------------------
    lofo_rows, val_unseen = [], {}
    for held in TYPES:
        log(f"E2: holding out {held}")
        known = [t for t in TYPES if t != held]
        mf = fit_ensemble(train, val, known, members, combiner)
        # validation recall on the held-out type at this fold's thresholds (used for model SELECTION only)
        vh = val[val.fault_type == held]
        for k, (s, p) in predict_all(mf, vh, train_clean, live_isolation_forest(vh, val_clean)).items():
            val_unseen.setdefault(k, []).append(float(p.mean()))
        p_test = predict_all(mf, test, train_clean, live_test)
        pred_rows.append(long_predictions(test, p_test, "E2_lofo", held))
        unseen_mask = (test.fault_type == held).to_numpy()
        known_mask = test.fault_type.isin(known).to_numpy()
        clean_mask = (test.is_fault == 0).to_numpy()
        for k, (s, p) in p_test.items():
            sub = clean_mask | unseen_mask
            r = {"held_out": held, "model": k, "unseen_recall": float(p[unseen_mask].mean()),
                 "unseen_n": int(unseen_mask.sum()), "known_recall": float(p[known_mask].mean()),
                 "fpr": float(p[clean_mask].mean()),
                 **{f"clean_vs_unseen_{kk}": vv for kk, vv in metrics(test.is_fault[sub], p[sub], None if s is None else s[sub]).items()
                    if kk in ("precision", "f1", "roc_auc", "pr_auc")}}
            lofo_rows.append(r)
    lofo = pd.DataFrame(lofo_rows)
    lofo.to_csv(OUT / "lofo_results.csv", index=False)

    # natural source anomalies in TEST (E1 models): reported, never used for any choice
    nat_test = natural[natural.split == "test"].reset_index(drop=True)
    nat_rows = []
    if len(nat_test):
        p_nat = predict_all(m1, nat_test, train_clean, live_isolation_forest(nat_test, test_clean))
        for k, (s_, p_) in p_nat.items():
            for i, r in nat_test.iterrows():
                nat_rows.append({"model": k, "pickup_date": r.pickup_date, "natural_anomaly": r.natural_anomaly,
                                 "trip_count": r.trip_count, "avg_trip_distance_miles": r.avg_trip_distance_miles,
                                 "flagged": int(p_[i]), "score": None if s_ is None else float(s_[i])})
    nat_df = pd.DataFrame(nat_rows)
    nat_df.to_csv(OUT / "natural_source_anomalies_test.csv", index=False)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_csv(OUT / "predictions.csv", index=False)

    # pooled known-vs-unseen per model
    kv = []
    for k in e1.model:
        lk = lofo[lofo.model == k]
        n_unseen = lk.unseen_n.sum()
        hits = (lk.unseen_recall * lk.unseen_n).sum()
        _, lo, hi = wilson_interval(int(round(hits)), int(n_unseen))
        row = e1[e1.model == k].iloc[0]
        _, klo, khi = wilson_interval(int(row.tp), int(row.n_pos))
        kv.append({"model": k, "known_detection_rate": row.recall, "known_ci95": f"[{klo:.3f}, {khi:.3f}]",
                   "unseen_detection_rate": hits / n_unseen, "unseen_ci95": f"[{lo:.3f}, {hi:.3f}]",
                   "lofo_mean_fpr": lk.fpr.mean(), "e1_fpr": row.fpr})
    kv = pd.DataFrame(kv)
    kv.to_csv(OUT / "known_vs_unseen.csv", index=False)

    # ---------------- validation-only model selection -------------------
    sel_rows = []
    for k in [c for c in e1.model if c in m1.thresholds or c == "isolation_forest_live"]:
        vf1, vfpr = val_e1[k]["f1"], val_e1[k]["fpr"]
        vu = float(np.mean(val_unseen[k]))
        sel_rows.append({"model": k, "val_f1_known": vf1, "val_fpr": vfpr, "val_unseen_recall_lofo": vu,
                         "selection_score": 0.5 * vf1 + 0.5 * vu, "eligible": vfpr <= 0.05 + 1e-9})
    sel = pd.DataFrame(sel_rows).sort_values("selection_score", ascending=False)
    sel.to_csv(OUT / "model_selection_validation.csv", index=False)
    chosen = sel[sel.eligible].iloc[0].model
    log(f"validation selection -> {chosen}\n{sel.to_string()}")
    # the artifact's strategy is the best candidate it can express; if the
    # existing live IsolationForest wins outright, it stays the default detector
    artifact_sel = sel[sel.eligible & (sel.model != "isolation_forest_live")].iloc[0].model
    default_detector = "isolation_forest" if chosen == "isolation_forest_live" else "hybrid"
    m1.strategy = artifact_sel

    # ---------------- latency -----------------------------------------
    lat = latency_per_row(m1, test)
    t0 = time.perf_counter(); live_isolation_forest(test.iloc[:30], test_clean)
    lat["isolation_forest_live"] = {"batch_ms_per_row": np.nan, "single_row_ms": (time.perf_counter() - t0) / 30 * 1e3}
    e1["batch_ms_per_row"] = e1.model.map(lambda k: lat.get(k, {}).get("batch_ms_per_row", np.nan))
    e1["single_row_ms"] = e1.model.map(lambda k: lat.get(k, {}).get("single_row_ms", np.nan))
    e1["operational_role"] = e1.model.map(lambda k: {
        "isolation_forest_live": "deployed baseline / fallback (ADIE_DETECTOR=isolation_forest)",
        artifact_sel: ("SELECTED operational detector (ADIE_DETECTOR=hybrid)" if default_detector == "hybrid"
                       else "hybrid artifact strategy (available via ADIE_DETECTOR=hybrid; not default)"),
    }.get(k, "research comparison only" if k in ("statistical_zscore", "isolation_forest_fit_once_legacy")
          or (k == "autoencoder" and "autoencoder" not in members)
          else "component of the hybrid artifact (scored + explained in the pipeline)"))
    e1.to_csv(OUT / "model_comparison.csv", index=False)

    # ---------------- explanations + localisation + RCA ----------------
    ex = explanations(m1, test, train, pred_df[pred_df.experiment == "E1_known"])
    # localisation method for the artifact: chosen on VALIDATION faults
    Xvf = to_matrix(val[val.is_fault == 1]); vexp = val[val.is_fault == 1].expected_feature.to_numpy()
    sx = A.tree_shap(m1.models["xgboost"], Xvf).values
    loc_val = {
        "zscore": np.mean(np.array(FEATURE_COLUMNS)[np.argmax(np.abs(m1.zscores(Xvf)), 1)] == vexp),
        "ecod": np.mean(np.array(FEATURE_COLUMNS)[np.argmax(A.ecod_contributions(m1.models["ecod"], Xvf), 1)] == vexp),
        "shap_supervised": np.mean(np.array(FEATURE_COLUMNS)[np.argmax(np.where(sx > 0, sx, 0), 1)] == vexp),
    }
    m1.localization = max(loc_val, key=lambda k: (loc_val[k], k == "zscore"))
    log(f"localization accuracy on VALIDATION faults: {loc_val} -> {m1.localization}")
    rca = rca_handoff(m1, test, pred_df[pred_df.experiment == "E1_known"], list(loc_val))

    # ---------------- figures -----------------------------------------
    fig_comparison(e1, FIG / "ensemble_vs_single_models.png")
    fig_lofo(lofo, FIG / "lofo_unseen_detection.png")
    fig_known_vs_unseen(kv, FIG / "known_vs_unseen.png")
    fig_severity(pred_df[pred_df.experiment == "E1_known"], FIG / "detection_by_severity.png")
    fig_scores(test, m1, FIG / "precision_recall_e1.png")

    # legit-event detail
    ev = pred_df[(pred_df.experiment == "E1_known") & (pred_df.legit_event != "") & (pred_df.y_true == 0)]
    ev.pivot_table(index=["pickup_date", "legit_event"], columns="model", values="y_pred").to_csv(OUT / "legit_event_alarms.csv")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": {s: {"start": d.pickup_date.min(), "end": d.pickup_date.max(), "rows": int(len(d)),
                       "clean_days": int((d.is_fault == 0).sum()), "fault_rows": int(d.is_fault.sum())}
                   for s, d in (("train", train), ("val", val), ("test", test))},
        "e0_unsupervised_design": {"members": members, "combiner": combiner, "autoencoder_evaluated": ae_available},
        "selected_by_validation": chosen, "artifact_strategy": artifact_sel, "default_detector": default_detector,
        "localization_validation_accuracy": loc_val, "localization_selected": m1.localization,
        "fusion": {"weight_sup": m1.weight, "t_sup": m1.t_sup, "t_unsup": m1.t_unsup, "thresholds": m1.thresholds},
        "natural_source_anomalies_held_aside": natural[["split", "pickup_date", "natural_anomaly"]].to_dict("records"),
        "natural_test_detection_rate": nat_df.groupby("model").flagged.mean().to_dict() if len(nat_df) else None,
        "missing_partition_completeness_recall": float(missing.completeness_detected.mean()) if len(missing) else None,
        "shap_vs_lime": {"top1_agreement": float(ex["shap_vs_lime"].top1_agree.mean()),
                         "mean_top3_jaccard": float(ex["shap_vs_lime"].top3_jaccard.mean()),
                         "mean_spearman_abs": float(ex["shap_vs_lime"].spearman_abs.mean()),
                         "n": int(len(ex["shap_vs_lime"]))},
        "rca_handoff_top1": rca.groupby("method").top1_hit.mean().to_dict() if rca is not None else None,
        "latency": lat,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (OUT / "selection.json").write_text(json.dumps({
        "strategy": artifact_sel, "default_detector": default_detector, "members": members, "combiner": combiner, "localization": m1.localization}, indent=2))

    cols = ["model", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "fpr_legit_event_days",
            "precision_at_5pct_prevalence", "single_row_ms"]
    log("\n" + e1[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    log("\n" + kv.to_string(float_format=lambda v: f"{v:.3f}"))
    log("DONE")


if __name__ == "__main__":
    main()
