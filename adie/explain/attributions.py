"""
Feature attributions for the hybrid ensemble -- every explainer is an
existing library (SHAP, LIME, PyOD's ECOD decomposition); this module only
picks the right interface for each model and normalises the output.

WHAT THESE EXPLANATIONS MEAN (and do not mean)
  * Supervised SHAP/LIME explain the *fault probability* p_sup: "which
    features pushed XGBoost/RandomForest towards 'this looks like a fault
    I was trained on'". SHAP values from TreeExplainer are in the model's
    raw output space (log-odds for XGBoost, probability for RandomForest).
  * Unsupervised attributions explain the *anomaly score*: "which features
    make this day unlike clean history". ECOD's are exact (its score is a
    sum of per-feature tail scores); IsolationForest's come from
    TreeExplainer on the isolation path length; the calibrated ensemble
    score is model-agnostic (permutation SHAP / LIME regression).
  * None of these establish the *causal origin* of a pipeline failure.
    They describe which of the five aggregate features moved. Which asset
    in the DAG caused it is decided separately by the lineage-constrained
    RCA scorer (adie/rca/scorer.py).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from adie.detection.isolation_forest import FEATURE_COLUMNS


def _top(values: np.ndarray, x: np.ndarray, k: int = 5, positive_first: bool = False) -> list[dict[str, Any]]:
    """Top-k features by |contribution| (with only five features, top-5 is
    the complete ranking)."""
    order = np.argsort(-np.abs(values))
    if positive_first:  # features pushing *towards* the alarm rank above those pushing away
        order = sorted(order, key=lambda j: (values[j] <= 0, -abs(values[j])))
    return [{"feature": FEATURE_COLUMNS[j], "value": float(x[j]), "contribution": float(values[j])}
            for j in list(order)[:k]]


# ---- SHAP -------------------------------------------------------------------

def tree_shap(model, X: np.ndarray):
    """shap.TreeExplainer for XGBoost / RandomForest / IsolationForest.
    Returns a shap.Explanation restricted to the positive ('fault') class
    for classifiers."""
    import shap

    exp = shap.TreeExplainer(model)(np.asarray(X, dtype=float))
    exp.feature_names = FEATURE_COLUMNS
    if exp.values.ndim == 3:  # sklearn RF: (n, features, classes)
        exp = exp[..., 1]
    return exp


def score_shap(fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, background: np.ndarray, seed: int = 7):
    """Model-agnostic SHAP (permutation algorithm) for any scalar score --
    used for the calibrated unsupervised score and the fused ensemble score,
    with clean TRAIN days as the background distribution."""
    import shap

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(fn, masker, algorithm="permutation", feature_names=FEATURE_COLUMNS, seed=seed)
    return explainer(np.asarray(X, dtype=float), max_evals=2 * len(FEATURE_COLUMNS) + 1 + 20, silent=True)


def ecod_contributions(model, X: np.ndarray) -> np.ndarray:
    from adie.detection.hybrid import ecod_raw

    _, contribs = ecod_raw(model, X, return_contributions=True)
    return contribs


# ---- LIME -------------------------------------------------------------------

def lime_explainer(X_train: np.ndarray, mode: str = "classification", seed: int = 7):
    """LimeTabularExplainer whose sampling statistics (quartile bins, means,
    stds) come from the TRAINING split only -- never validation/test."""
    from lime.lime_tabular import LimeTabularExplainer

    return LimeTabularExplainer(
        np.asarray(X_train, dtype=float), feature_names=FEATURE_COLUMNS,
        class_names=["clean", "fault"] if mode == "classification" else None,
        mode=mode, discretize_continuous=True, random_state=seed,
    )


def lime_explain(explainer, x: np.ndarray, predict_fn, num_features: int = 5, num_samples: int = 2000):
    """Returns (weights aligned with FEATURE_COLUMNS, readable rules, the
    lime Explanation). Classification -> predict_fn must be predict_proba
    (n, 2) and the 'fault' class (label 1) is explained; regression ->
    predict_fn returns a 1-D score."""
    if explainer.mode == "classification":
        exp = explainer.explain_instance(np.asarray(x, float), predict_fn, num_features=num_features,
                                         num_samples=num_samples, labels=(1,))
        pairs, rules = exp.as_map()[1], exp.as_list(label=1)
    else:
        exp = explainer.explain_instance(np.asarray(x, float), predict_fn, num_features=num_features,
                                         num_samples=num_samples)
        pairs, rules = exp.as_map()[next(iter(exp.as_map()))], exp.as_list()
    w = np.zeros(len(FEATURE_COLUMNS))
    for j, weight in pairs:
        w[j] = weight
    return w, rules, exp


# ---- one call used by the live detector --------------------------------------

def explain_rows(model, X: np.ndarray, with_lime: bool = True) -> dict[int, dict[str, Any]]:
    """Per flagged day: supervised SHAP top-5 (fault probability), ECOD
    top-5 (anomaly score), optional LIME top-5 (fault probability), and a
    `primary_top5` = whichever branch actually raised the alarm."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    sup = model.models[model.supervised_primary]
    sup_shap = tree_shap(sup, X).values
    ecod = ecod_contributions(model.models["ecod"], X) if "ecod" in model.models else None
    p = model.sup_proba(X)
    lime_exp = lime_explainer(model.background) if with_lime and model.background is not None else None
    # NOTE: background is a 100-day clean TRAIN sample; LIME only uses it for
    # per-feature sampling statistics.

    out: dict[int, dict[str, Any]] = {}
    for i, x in enumerate(X):
        d: dict[str, Any] = {
            "supervised_shap_top5": _top(sup_shap[i], x, positive_first=True),
            "supervised_shap_space": "log-odds" if model.supervised_primary == "xgboost" else "probability",
        }
        if ecod is not None:
            d["ecod_top5"] = _top(ecod[i], x)
        if lime_exp is not None:
            w, rules, _ = lime_explain(lime_exp, x, sup.predict_proba)
            d["lime_top5"] = _top(w, x, positive_first=True)
            d["lime_rules"] = [[r, float(v)] for r, v in rules]
        d["primary_top5"] = d["supervised_shap_top5"] if p[i] >= model.t_sup else d.get("ecod_top5", d["supervised_shap_top5"])
        d["primary_explains"] = "supervised fault probability" if p[i] >= model.t_sup else "unsupervised anomaly score"
        d["disclaimer"] = ("Feature attributions describe which aggregate features moved; they do not identify "
                           "the causal origin. Root cause is determined by lineage-based RCA.")
        out[i] = d
    return out


def save_waterfall(model, x: np.ndarray, path, title: str = "") -> str:
    """SHAP waterfall of the supervised fault probability (log-odds for
    XGBoost) for one day -- written by the Dagster explanation asset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    exp = tree_shap(model.models[model.supervised_primary], np.asarray(x, float)[None, :])
    one = shap.Explanation(values=exp.values[0], base_values=float(np.ravel(exp.base_values)[0]),
                           data=np.asarray(x, float), feature_names=FEATURE_COLUMNS)
    plt.figure()
    shap.plots.waterfall(one, show=False)
    plt.title(title, fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close("all")
    return str(path)
