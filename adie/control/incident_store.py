"""
Control plane: an append-only incident/audit log, backed by SQLite (stdlib).

Deliberately the simplest thing that satisfies "the engine must be able to
explain what it did and why": one row per lifecycle event, never updated in
place, queryable by incident id.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists incident_events (
    id integer primary key autoincrement,
    incident_id text not null,
    event_type text not null,
    payload text not null,
    created_at text not null default (datetime('now'))
);
create index if not exists idx_incident_id on incident_events(incident_id);
"""


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


class IncidentStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA)
            con.commit()
        finally:
            con.close()

    def record(self, incident_id: str, event_type: str, payload: Any) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.execute(
                "insert into incident_events (incident_id, event_type, payload) values (?, ?, ?)",
                (incident_id, event_type, json.dumps(payload, default=_json_default)),
            )
            con.commit()
        finally:
            con.close()

    def timeline(self, incident_id: str) -> list[dict]:
        con = sqlite3.connect(self.db_path)
        try:
            rows = con.execute(
                "select event_type, payload, created_at from incident_events "
                "where incident_id = ? order by id asc",
                (incident_id,),
            ).fetchall()
        finally:
            con.close()
        return [
            {"event_type": r[0], "payload": json.loads(r[1]), "created_at": r[2]}
            for r in rows
        ]

    def all_incident_ids(self) -> list[str]:
        con = sqlite3.connect(self.db_path)
        try:
            rows = con.execute(
                "select distinct incident_id from incident_events order by incident_id"
            ).fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]
