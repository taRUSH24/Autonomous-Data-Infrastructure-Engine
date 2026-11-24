"""
Shared envelope every plane speaks: Signal -> Hypothesis -> Action -> Verification.

Kept intentionally small and dependency-light (stdlib dataclasses only) so every
plane can import it without pulling in Dagster, dbt, or LangGraph.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_day(day: str) -> str:
    """Shared guard for every call site that interpolates a day string
    directly into SQL (adie.faults.inject, adie.agent.tools) rather than a
    bound parameter -- DuckDB's Python API only binds `?` in value position,
    not cleanly inside every `::date` cast shape these fault/repair
    statements use. In the live agent path `day` can originate from an
    LLM's structured-output response (adie.agent.graph), so this is a real
    boundary, not a theoretical one: reject anything that isn't a plain
    YYYY-MM-DD before it ever reaches an f-string bound for production SQL."""
    if not _DAY_RE.match(day):
        raise ValueError(f"day must be in YYYY-MM-DD format, got {day!r}")
    return day


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    CRITICAL = "critical"


class CheckKind(str, Enum):
    RULE = "rule"          # dbt test / asset check
    ANOMALY = "anomaly"    # statistical / ML detector


@dataclass
class Signal:
    """One detection event emitted by plane 1 (Detection)."""
    asset_key: str
    check_name: str
    kind: CheckKind
    severity: Severity
    metric: str
    observed: float | None = None
    expected: float | None = None
    message: str = ""
    column: str | None = None
    detected_at: str = field(default_factory=utcnow)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["kind"] = self.kind.value
        d["severity"] = self.severity.value
        return d


@dataclass
class Hypothesis:
    """One ranked root-cause candidate emitted by plane 3 (RCA)."""
    asset_key: str
    mechanism: str              # e.g. "upstream_freshness", "schema_drift", "join_fanout"
    score: float                # higher = more likely root cause
    evidence: list[str] = field(default_factory=list)
    distance_from_signal: int = 0  # hops in the upstream cone
    supporting_signals: list[str] = field(default_factory=list)  # check_names
    implicating_days: list[str] = field(default_factory=list)  # pickup_dates whose metric specifically matched this mechanism

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ActionKind(str, Enum):
    REMATERIALIZE = "rematerialize"
    SQL_PATCH = "sql_patch"
    NO_OP = "no_op"


@dataclass
class Action:
    """One proposed repair emitted by plane 4 (Agent)."""
    kind: ActionKind
    target_asset: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)  # e.g. {"sql": "..."}
    proposed_at: str = field(default_factory=utcnow)
    approved: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["kind"] = self.kind.value
        return d


@dataclass
class Verification:
    """Result of running an Action against a sandboxed clone, never production."""
    action: Action
    passed: bool
    checks_before: dict[str, Any] = field(default_factory=dict)
    checks_after: dict[str, Any] = field(default_factory=dict)
    sandbox_db_path: str = ""
    log: list[str] = field(default_factory=list)
    verified_at: str = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["action"] = self.action.to_dict()
        return d
