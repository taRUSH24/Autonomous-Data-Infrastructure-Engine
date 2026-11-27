#!/usr/bin/env python3
"""
Trains the OPERATIONAL hybrid-ensemble artifact the Dagster pipeline loads
(models/hybrid_ensemble/hybrid_ensemble.joblib + metadata.json), so the
pipeline never retrains on each run.

It is produced by the same code and data protocol that was evaluated
(adie.eval.hybrid_experiment.fit_ensemble): fit on TRAIN (2023-01-15 ..
06-30), calibrated on VALIDATION (07-01 .. 08-31), all five row-level fault
types known to the supervised branch. Strategy / unsupervised members /
combiner / localization method come from experiments/results/hybrid/
selection.json -- a choice made on validation only by
scripts/run_hybrid_experiments.py. TEST data is never touched here.

Run: .venv/bin/python scripts/train_hybrid_ensemble.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from adie.detection.hybrid import DEFAULT_ARTIFACT_PATH  # noqa: E402
from adie.eval import hybrid_data  # noqa: E402
from adie.eval.hybrid_experiment import fit_ensemble  # noqa: E402

SELECTION = REPO_ROOT / "experiments" / "results" / "hybrid" / "selection.json"


def main() -> None:
    sel = json.loads(SELECTION.read_text()) if SELECTION.exists() else {
        "strategy": "hybrid_gated", "members": ["isolation_forest", "ecod"], "combiner": "average",
        "localization": "zscore", "default_detector": "hybrid"}
    feats, _ = hybrid_data.build_or_load()
    # same protocol as the evaluation: real source incidents are not "clean"
    feats = feats[~((feats.is_fault == 0) & (feats.natural_anomaly != ""))]
    train, val = feats[feats.split == "train"], feats[feats.split == "val"]

    m = fit_ensemble(train, val, hybrid_data.ROW_FAULT_TYPES, sel["members"], sel["combiner"])
    m.strategy = sel["strategy"]
    m.localization = sel["localization"]
    import pyod, shap, sklearn, xgboost

    m.metadata.update({
        "trained_on": f"{train.pickup_date.min()}..{train.pickup_date.max()}",
        "calibrated_on": f"{val.pickup_date.min()}..{val.pickup_date.max()}",
        "selection_source": str(SELECTION.relative_to(REPO_ROOT)) if SELECTION.exists() else "defaults",
        "recommended_default_detector": sel.get("default_detector", "hybrid"),
        "library_versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
                             "pyod": pyod.__version__, "shap": shap.__version__},
    })
    path = m.save(DEFAULT_ARTIFACT_PATH)
    print(f"saved {path} ({path.stat().st_size / 1e6:.2f} MB): strategy={m.strategy}, "
          f"members={m.unsup_members}, weight={m.weight:.2f}, t_sup={m.t_sup:.3f}, t_unsup={m.t_unsup:.3f}")


if __name__ == "__main__":
    main()
