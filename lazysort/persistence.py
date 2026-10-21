from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class HistoryStore:
    def __init__(self, database: Path):
        self.database = database
        try:
            self.database.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise OSError(f"Cannot create LazySort history in {self.database.parent}: {exc}") from exc
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    undone_at TEXT
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def record(self, source: Path, destination: Path) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO operations(source, destination, created_at) VALUES (?, ?, ?)",
                (str(source), str(destination), datetime.now(timezone.utc).isoformat()),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[tuple]:
        with self._connect() as connection:
            return connection.execute(
                "SELECT id, source, destination, created_at, undone_at "
                "FROM operations ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def mark_undone(self, operation_id: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE operations SET undone_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), operation_id),
            )
