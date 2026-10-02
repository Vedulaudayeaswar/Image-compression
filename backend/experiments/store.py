from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import DATA_DIR


DATABASE = DATA_DIR / "experiments.sqlite3"


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, filename TEXT NOT NULL,
        config TEXT NOT NULL, result TEXT NOT NULL)
    """)
    return connection


def save(experiment_id: str, filename: str, config: dict[str, Any], result: dict[str, Any]) -> None:
    with _connection() as db:
        db.execute("INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?)", (experiment_id, datetime.now(timezone.utc).isoformat(), filename, json.dumps(config), json.dumps(result)))


def list_experiments() -> list[dict[str, Any]]:
    with _connection() as db:
        rows = db.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [{"id": r["id"], "created_at": r["created_at"], "filename": r["filename"], "config": json.loads(r["config"]), "result": json.loads(r["result"])} for r in rows]


def get(experiment_id: str) -> dict[str, Any] | None:
    with _connection() as db: row = db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
    if row is None: return None
    return {"id": row["id"], "created_at": row["created_at"], "filename": row["filename"], "config": json.loads(row["config"]), "result": json.loads(row["result"])}
