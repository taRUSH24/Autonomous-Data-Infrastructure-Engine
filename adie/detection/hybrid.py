"""
Hybrid ensemble anomaly detector for fct_trips_daily (plane 1).

Two complementary branches, every model an existing library implementation:

  unsupervised  scikit-learn IsolationForest (same estimator the existing
                detector uses) + PyOD ECOD, both fit ONLY on clean historical
                days. They model "normal" and can flag observations unlike
                anything seen before -- but there is no guarantee they flag
                every unfamiliar fault (a fault that keeps the five features
                inside their normal range is invisible to them).
  supervised    XGBoost (primary) and RandomForest (comparison baseline),
                same hyper-parameters as adie/eval/models.py, trained on
                clean days + labelled injected faults. They recognise fault
                *patterns* they were shown.

What is written here (the only custom code): score calibration, the
fusion rule, and artifact save/load.

  * Calibration. Each unsupervised raw score is mapped to its empirical CDF
    position among *clean validation* days (a conformal-style score in
    [0, 1); beyond the largest clean score an exponential tail keeps the
    score monotone so severe anomalies still rank above mild ones). The two
    calibrated scores are averaged (or max-ed -- chosen on validation).
  * Fusion, two transparent strategies, all thresholds fit on validation:
      weighted  s = w * p_sup + (1 - w) * u_unsup ;  alarm if s >= t
      gated     alarm if p_sup >= t_sup  (known-fault pattern)
                   or u_unsup >= t_unsup (novel-anomaly gate, t_unsup set by
                   a strict false-positive budget on clean validation days)
                exposed as one continuous score g = max(p/t_sup, u/t_unsup),
                alarm iff g >= 1.
    The gate is what keeps "either branch fires" from flooding the agent
    with false positives: the unsupervised branch can only raise an alarm
    on its own when the day is more extreme than (nearly) every clean
    validation day.

The live Dagster pipeline loads a pre-trained artifact (see
scripts/train_hybrid_ensemble.py) instead of refitting per run. The
original per-batch IsolationForest detector (isolation_forest.py) is
untouched and remains the baseline/fallback -- see adie/detection/selector.py.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adie.contracts import CheckKind, Severity, Signal
from adie.detection.isolation_forest import ASSET_KEY, FEATURE_COLUMNS, load_daily_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "models" / "hybrid_ensemble"
DEFAULT_ARTIFACT_PATH = DEFAULT_ARTIFACT_DIR / "hybrid_ensemble.joblib"
CHECK_NAME = "hybrid_ensemble_daily_metrics"


def to_matrix(df: pd.DataFrame) -> np.ndarray:
    """Identical preprocessing to the existing detectors (adie/eval/models.py
    `_X`, isolation_forest.score_daily_metrics): the five features, NaN -> 0.
    A nulled-out column therefore shows up as an extreme 0, which is itself
    the anomaly signature for schema_change."""
    return df[FEATURE_COLUMNS].astype(float).fillna(0.0).to_numpy()


# ---------------------------------------------------------------------------
# unsupervised members (thin wrappers over library estimators)
# ---------------------------------------------------------------------------

def fit_isolation_forest(X_clean: np.ndarray, random_state: int = 7):
    from sklearn.ensemble import IsolationForest

    # contamination only sets sklearn's internal offset/predict(); we use the
    # continuous score_samples() and calibrate the threshold on validation.
    return IsolationForest(n_estimators=200, contamination=0.1, random_state=random_state).fit(X_clean)


def isolation_forest_raw(model, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)  # higher = more anomalous


def fit_ecod(X_clean: np.ndarray):
    from pyod.models.ecod import ECOD

    return ECOD().fit(X_clean)


def ecod_raw(model, X: np.ndarray, return_contributions: bool = False):
    """PyOD's ECOD.decision_function concatenates the query batch with the
    training set before computing the empirical CDFs, so scoring a batch
    jointly lets test points shift each other's scores (a test-time leak).
    Scoring one row at a time keeps every score relative to the training
    data only. ECOD's per-dimension tail scores (`model.O`) sum exactly to
    the decision score -- returned as an intrinsic feature attribution."""
    X = np.atleast_2d(X)
    scores = np.empty(len(X))
    contribs = np.empty_like(X, dtype=float)
    for i in range(len(X)):
        scores[i] = model.decision_function(X[i : i + 1])[0]
        contribs[i] = model.O[-1]
    return (scores, contribs) if return_contributions else scores


def fit_autoencoder(X_clean: np.ndarray, random_state: int = 7):
    """Optional third member (PyOD AutoEncoder, needs torch). Evaluated in
    the research harness; only included in the ensemble if it improves
    validation performance -- see scripts/run_hybrid_experiments.py."""
    from pyod.models.auto_encoder import AutoEncoder
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_clean)
    ae = AutoEncoder(hidden_neuron_list=[4, 2, 4], epoch_num=200, batch_size=16,
                     random_state=random_state, verbose=0, preprocessing=False)
    ae.fit(scaler.transform(X_clean))
    return (scaler, ae)


def autoencoder_raw(model, X: np.ndarray) -> np.ndarray:
    scaler, ae = model
    return ae.decision_function(scaler.transform(X))


RAW_SCORERS = {
    "isolation_forest": isolation_forest_raw,
    "ecod": ecod_raw,
    "autoencoder": autoencoder_raw,
}


@dataclass
class TailCalibrator:
    """Empirical-CDF calibration of a raw anomaly score against clean
    reference scores (clean *validation* days -- never the fit data, never
    test). u(s) = #{ref < s} / (n + 1); above max(ref) an exponential tail
    adds up to 1/(n+1) so u stays strictly monotone and < 1."""
    ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: float = 1.0

    def fit(self, raw_clean_ref: np.ndarray) -> "TailCalibrator":
        self.ref = np.sort(np.asarray(raw_clean_ref, dtype=float))
        s = float(np.std(self.ref))
        self.scale = s if s > 0 else 1.0
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        n = len(self.ref)
        u = np.searchsorted(self.ref, raw, side="left") / (n + 1)
        top = self.ref[-1]
        excess = np.clip(raw - top, 0, None)
        return u + (excess > 0) * (1 - np.exp(-excess / self.scale)) / (n + 1)


# ---------------------------------------------------------------------------
# the ensemble
# ---------------------------------------------------------------------------

@dataclass
class HybridEnsemble:
    unsup_members: list[str] = field(default_factory=lambda: ["isolation_forest", "ecod"])
    combiner: str = "average"            # "average" | "max" (chosen on validation)
    strategy: str = "hybrid_gated"        # any key of candidate_scores(): hybrid_gated, hybrid_weighted, xgboost, ...
    weight: float = 0.5                   # weighted strategy: w * p_sup + (1-w) * u
    t_sup: float = 0.5
    t_unsup: float = 0.99
    supervised_primary: str = "xgboost"
    localization: str = "zscore"          # which explanation names the Signal.metric (chosen on validation)
    # validation-calibrated decision threshold for every candidate score
    # (standalone members, supervised models, both fusion rules); `strategy`
    # may name any of them -- that is how a stronger standalone baseline is
    # preserved if it beats the fusion on validation.
    thresholds: dict[str, float] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    calibrators: dict[str, TailCalibrator] = field(default_factory=dict)
    background: np.ndarray | None = None  # clean TRAIN sample for SHAP/LIME
    train_mean: np.ndarray | None = None
    train_std: np.ndarray | None = None
    train_median: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- fitting -----------------------------------------------------------
    def fit_unsupervised(self, X_clean_train: np.ndarray, X_clean_cal: np.ndarray, random_state: int = 7):
        fitters = {"isolation_forest": lambda X: fit_isolation_forest(X, random_state),
                   "ecod": fit_ecod,
                   "autoencoder": lambda X: fit_autoencoder(X, random_state)}
        for name in self.unsup_members:
            self.models[name] = fitters[name](X_clean_train)
            self.calibrators[name] = TailCalibrator().fit(RAW_SCORERS[name](self.models[name], X_clean_cal))
        self.train_mean = X_clean_train.mean(axis=0)
        sd = X_clean_train.std(axis=0)
        sd[sd == 0] = 1.0
        self.train_std = sd
        self.train_median = np.median(X_clean_train, axis=0)
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_clean_train), size=min(100, len(X_clean_train)), replace=False)
        self.background = X_clean_train[idx]
        return self

    def fit_supervised(self, X: np.ndarray, y: np.ndarray, random_state: int = 7):
        from adie.eval.models import build_random_forest, build_xgboost

        self.models["xgboost"] = build_xgboost(y, random_state).fit(X, y)
        self.models["random_forest"] = build_random_forest(random_state).fit(X, y)
        return self

    # ---- scoring -----------------------------------------------------------
    def member_scores(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {n: self.calibrators[n].transform(RAW_SCORERS[n](self.models[n], X)) for n in self.unsup_members}

    def unsup_score(self, X: np.ndarray, members: dict[str, np.ndarray] | None = None) -> np.ndarray:
        m = np.column_stack(list((members or self.member_scores(X)).values()))
        return m.max(axis=1) if self.combiner == "max" else m.mean(axis=1)

    def sup_proba(self, X: np.ndarray, model: str | None = None) -> np.ndarray:
        return self.models[model or self.supervised_primary].predict_proba(X)[:, 1]

    def fuse_weighted(self, p_sup: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.weight * p_sup + (1 - self.weight) * u

    def fuse_gated(self, p_sup: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.maximum(p_sup / self.t_sup, u / self.t_unsup)

    def candidate_scores(self, X: np.ndarray, members: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Every candidate detector's continuous score (higher = more
        anomalous). Keys are the names used in thresholds / strategy."""
        members = members or self.member_scores(X)
        u = self.unsup_score(X, members)
        p = self.sup_proba(X)
        out = {n: v for n, v in members.items()}
        out["unsup_ensemble"] = u
        for n in ("xgboost", "random_forest"):
            if n in self.models:
                out[n] = self.sup_proba(X, n)
        out["hybrid_weighted"] = self.fuse_weighted(p, u)
        out["hybrid_gated"] = self.fuse_gated(p, u)
        return out

    def threshold(self, strategy: str | None = None) -> float:
        strategy = self._key(strategy or self.strategy)
        return 1.0 if strategy == "hybrid_gated" else self.thresholds[strategy]

    @staticmethod
    def _key(strategy: str) -> str:
        return {"gated": "hybrid_gated", "weighted": "hybrid_weighted"}.get(strategy, strategy)

    def score_fn(self, strategy: str | None = None):
        """Continuous score of one candidate as a function of a raw feature
        matrix -- the prediction interface handed to model-agnostic
        explainers (SHAP permutation, LIME regression)."""
        key = self._key(strategy or self.strategy)

        def f(X):
            return self.candidate_scores(np.asarray(X, dtype=float))[key]
        return f

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        X = to_matrix(df)
        t0 = time.perf_counter()
        members = self.member_scores(X)
        cands = self.candidate_scores(X, members)
        u, p = cands["unsup_ensemble"], self.sup_proba(X)
        s = cands[self._key(self.strategy)]
        latency = time.perf_counter() - t0
        out = df.copy()
        for n, v in members.items():
            out[f"u_{n}"] = v
        out["u_unsup"] = u
        out["p_sup"] = p
        out["ensemble_score"] = s
        out["is_anomaly"] = s >= self.threshold()
        known = p >= self.t_sup
        novel = u >= self.t_unsup
        out["branch"] = np.select([known & novel, known, novel], ["both", "supervised_known_fault",
                                                                    "unsupervised_novel"], default="none")
        if self._key(self.strategy) != "hybrid_gated":
            # branch labels describe the two gates; for any other strategy an
            # alarm neither gate would raise on its own is labelled by strategy
            out.loc[out["is_anomaly"] & (out["branch"] == "none"), "branch"] = self._key(self.strategy)
        out.attrs["latency_seconds"] = latency
        return out

    # ---- localization (which feature drove it) ----------------------------
    def zscores(self, X: np.ndarray) -> np.ndarray:
        return (X - self.train_mean) / self.train_std

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_ARTIFACT_PATH) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        (path.parent / "metadata.json").write_text(json.dumps(self.describe(), indent=2, default=str))
        return path

    def describe(self) -> dict[str, Any]:
        return {
            "features": FEATURE_COLUMNS,
            "unsup_members": self.unsup_members, "combiner": self.combiner,
            "strategy": self.strategy, "weight": self.weight,
            "t_sup": self.t_sup, "t_unsup": self.t_unsup, "thresholds": self.thresholds,
            "supervised_primary": self.supervised_primary, "localization": self.localization,
            **self.metadata,
        }


_CACHE: dict[str, HybridEnsemble] = {}


def load(path: str | Path = DEFAULT_ARTIFACT_PATH) -> HybridEnsemble:
    import joblib

    key = str(Path(path).resolve())
    if key not in _CACHE:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"hybrid ensemble artifact not found at {path} -- run scripts/train_hybrid_ensemble.py"
            )
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


# ---------------------------------------------------------------------------
# production entry point: same shape as isolation_forest.detect()
# ---------------------------------------------------------------------------

def _localize(model: HybridEnsemble, x: np.ndarray, branch: str, expl: dict | None) -> str:
    """Name the single feature most responsible -- the hint RCA's
    MECHANISM_BY_METRIC table consumes. Method chosen on validation
    (model.localization). This is a *localization hint*, not a causal claim;
    causal attribution to an asset stays with the lineage-based RCA."""
    method = model.localization
    if expl and method == "shap_supervised" and branch not in ("unsupervised_novel",):
        return expl["supervised_shap_top5"][0]["feature"]
    if expl and method in ("ecod", "shap_supervised"):
        return expl["ecod_top5"][0]["feature"]
    z = np.abs(model.zscores(x[None, :])[0])
    return FEATURE_COLUMNS[int(np.argmax(z))]


def detect(duckdb_path: str, artifact_path: str | Path | None = None, explain: bool = True,
           with_lime: bool = True) -> list[Signal]:
    model = load(artifact_path or DEFAULT_ARTIFACT_PATH)
    df = load_daily_metrics(duckdb_path)
    if df.empty:
        return []
    scored = model.predict_frame(df)
    flagged = scored[scored["is_anomaly"]]
    if flagged.empty:
        return []

    explanations: dict[int, dict] = {}
    if explain:
        from adie.explain.attributions import explain_rows

        explanations = explain_rows(model, to_matrix(flagged), with_lime=with_lime)

    signals = []
    X_flag = to_matrix(flagged)
    for i, (_, row) in enumerate(flagged.iterrows()):
        x = X_flag[i]
        expl = explanations.get(i)
        metric = _localize(model, x, row["branch"], expl)
        j = FEATURE_COLUMNS.index(metric)
        direction = "above" if x[j] > model.train_median[j] else "below"
        severity = Severity.CRITICAL if (row["branch"] == "both" or row["p_sup"] >= 0.9) else Severity.WARN
        day = str(pd.Timestamp(row["pickup_date"]).date())
        top = ", ".join(f"{d['feature']} ({d['contribution']:+.3f})" for d in (expl or {}).get("primary_top5", [])[:3])
        signals.append(Signal(
            asset_key=ASSET_KEY,
            check_name=CHECK_NAME,
            kind=CheckKind.ANOMALY,
            severity=severity,
            metric=metric,
            observed=float(x[j]),
            expected=float(model.train_median[j]),
            message=(
                f"{day} flagged by hybrid ensemble ({row['branch']}; score={row['ensemble_score']:.3f}, "
                f"p_fault={row['p_sup']:.3f}, unsup={row['u_unsup']:.3f}); localized to {metric} "
                f"({direction} normal: {x[j]:.2f} vs median {model.train_median[j]:.2f})"
                + (f"; top attributions: {top}" if top else "")
            ),
            raw={
                "pickup_date": day,
                **{c: float(row[c]) if pd.notna(row[c]) else None for c in FEATURE_COLUMNS},
                "branch": row["branch"], "direction": direction,
                "p_sup": float(row["p_sup"]), "u_unsup": float(row["u_unsup"]),
                "ensemble_score": float(row["ensemble_score"]),
                "explanation": expl or {},
            },
        ))
    return signals
