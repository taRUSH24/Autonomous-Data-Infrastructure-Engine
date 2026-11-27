"""
Fitting + validation calibration + evaluation for the hybrid ensemble,
shared by scripts/run_hybrid_experiments.py (research) and
scripts/train_hybrid_ensemble.py (operational artifact) so the deployed
model is produced by exactly the code that was evaluated.

Leakage rules enforced here:
  * unsupervised models are fit on TRAIN clean days only;
  * score calibration, every threshold, and the fusion weight use VALIDATION
    rows only -- and, in leave-one-fault-type-out folds, only validation rows
    of the fault types the supervised models were trained on;
  * TEST rows are only ever passed to `evaluate`.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from adie.detection.hybrid import HybridEnsemble, to_matrix

MAX_FPR = 0.05         # every calibrated threshold must keep validation FPR <= 5%
ALPHA_UNSUP_GATE = 0.02  # the gated rule's unsupervised-only alarm: <= 2% of clean validation days
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

CANDIDATES = ["isolation_forest", "ecod", "autoencoder", "unsup_ensemble", "xgboost", "random_forest",
              "hybrid_weighted", "hybrid_gated"]


def _midpoints(scores: np.ndarray) -> np.ndarray:
    u = np.unique(scores)
    mids = (u[1:] + u[:-1]) / 2
    return np.concatenate([[u[0] - 1e-9], mids, [u[-1] + 1e-9]])


def _rates(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    neg = int((y == 0).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1, (fp / neg if neg else 0.0)


def choose_threshold(scores: np.ndarray, y: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """Max validation F1 subject to validation FPR <= max_fpr (ties -> lower
    FPR, then higher threshold). Thresholds sit at midpoints between
    observed scores, never exactly on a validation point."""
    best, best_key = None, None
    for t in _midpoints(scores):
        f1, fpr = _rates(scores >= t, y)
        if fpr > max_fpr:
            continue
        key = (f1, -fpr, t)
        if best_key is None or key > best_key:
            best, best_key = t, key
    return float(best if best is not None else scores.max() + 1e-9)


def fpr_threshold(clean_scores: np.ndarray, alpha: float) -> float:
    """Label-free threshold: at most floor(alpha * n) clean days above it."""
    s = np.sort(np.asarray(clean_scores))[::-1]
    k = int(np.floor(alpha * len(s)))
    if k == 0:
        return float(s[0] + 1e-9)
    return float((s[k - 1] + s[k]) / 2)


def fit_ensemble(train: pd.DataFrame, val: pd.DataFrame, sup_fault_types: list[str],
                 members: list[str], combiner: str = "average", random_state: int = 7) -> HybridEnsemble:
    """Fits both branches and calibrates every candidate's threshold plus the
    two fusion rules on validation rows of `sup_fault_types` (+ clean)."""
    m = HybridEnsemble(unsup_members=list(members), combiner=combiner)
    tr_clean, va_clean = train[train.is_fault == 0], val[val.is_fault == 0]
    m.fit_unsupervised(to_matrix(tr_clean), to_matrix(va_clean), random_state)

    tr_sup = train[(train.is_fault == 0) | train.fault_type.isin(sup_fault_types)]
    m.fit_supervised(to_matrix(tr_sup), tr_sup.is_fault.to_numpy(), random_state)

    va = val[(val.is_fault == 0) | val.fault_type.isin(sup_fault_types)]
    Xv, yv = to_matrix(va), va.is_fault.to_numpy()
    members_v = m.member_scores(Xv)
    u, p = m.unsup_score(Xv, members_v), m.sup_proba(Xv)

    # gated: novelty gate from a label-free clean-FPR budget, then the
    # supervised threshold that maximises F1 with total FPR <= MAX_FPR
    m.t_unsup = fpr_threshold(m.unsup_score(to_matrix(va_clean)), ALPHA_UNSUP_GATE)
    gate = u >= m.t_unsup
    best = None
    for t in _midpoints(p):
        f1, fpr = _rates((p >= t) | gate, yv)
        if fpr <= MAX_FPR and (best is None or (f1, -fpr, t) > best[0]):
            best = ((f1, -fpr, t), t)
    m.t_sup = float(best[1] if best else min(p.max() + 1e-9, 1.0))

    # weighted: grid-search w, threshold per w, keep best validation F1
    best = None
    for w in WEIGHT_GRID:
        s = w * p + (1 - w) * u
        t = choose_threshold(s, yv)
        f1, fpr = _rates(s >= t, yv)
        ap = average_precision_score(yv, s) if 0 < yv.sum() < len(yv) else 0.0
        key = (round(f1, 6), round(ap, 6), -abs(w - 0.5))
        if best is None or key > best[0]:
            best = (key, w, t)
    m.weight = float(best[1])

    cands = m.candidate_scores(Xv, members_v)
    m.thresholds = {k: choose_threshold(v, yv) for k, v in cands.items() if k != "hybrid_gated"}
    m.thresholds["hybrid_weighted"] = float(best[2])
    m.thresholds["hybrid_gated"] = 1.0
    m.metadata.update({
        "sup_fault_types": list(sup_fault_types),
        "n_train_clean": int(len(tr_clean)), "n_train_fault_rows": int(tr_sup.is_fault.sum()),
        "n_val_clean": int(len(va_clean)), "n_val_fault_rows": int(va.is_fault.sum()),
        "max_val_fpr": MAX_FPR, "unsup_gate_alpha": ALPHA_UNSUP_GATE,
    })
    return m


def candidate_predictions(m: HybridEnsemble, df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (score, binary prediction at the validation-calibrated threshold)."""
    cands = m.candidate_scores(to_matrix(df))
    return {k: (v, v >= m.threshold(k)) for k, v in cands.items()}


def latency_per_row(m: HybridEnsemble, df: pd.DataFrame, n_single: int = 30) -> dict[str, dict[str, float]]:
    """Batch and single-row scoring latency per candidate (wall clock, this machine)."""
    X = to_matrix(df)
    out: dict[str, dict[str, float]] = {}
    fns = {
        "isolation_forest": lambda X: m.member_scores(X)["isolation_forest"],
        "ecod": lambda X: m.member_scores(X)["ecod"],
        "xgboost": lambda X: m.sup_proba(X, "xgboost"),
        "random_forest": lambda X: m.sup_proba(X, "random_forest"),
        "unsup_ensemble": lambda X: m.unsup_score(X),
        "hybrid_weighted": lambda X: m.candidate_scores(X)["hybrid_weighted"],
        "hybrid_gated": lambda X: m.candidate_scores(X)["hybrid_gated"],
    }
    if "autoencoder" in m.unsup_members:
        fns["autoencoder"] = lambda X: m.member_scores(X)["autoencoder"]
    for name, fn in fns.items():
        t0 = time.perf_counter(); fn(X); batch = (time.perf_counter() - t0) / len(X)
        t0 = time.perf_counter()
        for i in range(min(n_single, len(X))):
            fn(X[i : i + 1])
        single = (time.perf_counter() - t0) / min(n_single, len(X))
        out[name] = {"batch_ms_per_row": batch * 1e3, "single_row_ms": single * 1e3}
    return out


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None, prevalence: float = 0.05) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": prec, "recall": rec, "f1": f1, "fpr": fpr}
    # precision if faults were as rare as `prevalence` of days (the test set is
    # ~50% faults by construction, which flatters precision)
    if not np.isnan(rec) and not np.isnan(fpr) and (rec * prevalence + fpr * (1 - prevalence)) > 0:
        out[f"precision_at_{int(prevalence * 100)}pct_prevalence"] = rec * prevalence / (
            rec * prevalence + fpr * (1 - prevalence))
    if score is not None and 0 < y.sum() < len(y):
        s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out
