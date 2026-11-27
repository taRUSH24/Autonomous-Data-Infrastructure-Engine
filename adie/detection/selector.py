"""
Configuration switch between the original detector and the hybrid ensemble.

    ADIE_DETECTOR=isolation_forest   original per-batch IsolationForest (baseline)
    ADIE_DETECTOR=hybrid             pre-trained hybrid ensemble (adie/detection/hybrid.py)
    (unset)                          the artifact's validation-selected recommendation
                                     (models/hybrid_ensemble/metadata.json), else isolation_forest
    ADIE_HYBRID_ARTIFACT=<path>      override the artifact location

If the hybrid artifact is missing or fails to load/score, detection falls
back to the original IsolationForest and says so -- the pipeline never
silently runs without a row-level detector. The calendar-completeness rule
check runs alongside either one (neither ML detector can see a day that
has no row).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from adie.contracts import Signal
from adie.detection import completeness, isolation_forest

logger = logging.getLogger(__name__)
VALID = ("isolation_forest", "hybrid")


def artifact_path() -> Path:
    from adie.detection.hybrid import DEFAULT_ARTIFACT_PATH

    return Path(os.environ.get("ADIE_HYBRID_ARTIFACT", DEFAULT_ARTIFACT_PATH))


def configured_detector() -> str:
    name = os.environ.get("ADIE_DETECTOR", "").strip().lower()
    if name:
        if name not in VALID:
            raise ValueError(f"ADIE_DETECTOR must be one of {VALID}, got {name!r}")
        return name
    meta = artifact_path().parent / "metadata.json"
    if meta.exists():
        rec = json.loads(meta.read_text()).get("recommended_default_detector")
        if rec in VALID:
            return rec
    return "isolation_forest"


def detect_row_level(duckdb_path: str, name: str | None = None, explain: bool = True) -> tuple[list[Signal], str]:
    """Returns (signals, detector_actually_used)."""
    name = name or configured_detector()
    if name == "hybrid":
        try:
            from adie.detection import hybrid

            return hybrid.detect(duckdb_path, artifact_path=artifact_path(), explain=explain), "hybrid"
        except Exception as e:  # missing artifact, version mismatch, ...
            logger.warning("hybrid detector unavailable (%s: %s); falling back to isolation_forest",
                           type(e).__name__, e)
            return isolation_forest.detect(duckdb_path), "isolation_forest (fallback)"
    return isolation_forest.detect(duckdb_path), "isolation_forest"


def detect_all(duckdb_path: str, name: str | None = None, explain: bool = True) -> tuple[list[Signal], str]:
    signals, used = detect_row_level(duckdb_path, name, explain)
    return signals + completeness.detect(duckdb_path), used
