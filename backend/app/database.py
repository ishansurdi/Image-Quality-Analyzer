"""Small SQLite persistence layer for image analysis results."""

import json
import sqlite3
from pathlib import Path
from typing import Any


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    quality_label TEXT NOT NULL,
    issues_json TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""


class AnalysisRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(CREATE_TABLE)

    def create(self, analysis: dict[str, Any]) -> dict[str, Any]:
        query = """
            INSERT INTO analyses (
                original_filename, stored_filename, content_type, width, height,
                quality_score, quality_label, issues_json, statistics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = (
            analysis["original_filename"],
            analysis["stored_filename"],
            analysis["content_type"],
            analysis["width"],
            analysis["height"],
            analysis["quality_score"],
            analysis["quality_label"],
            json.dumps(analysis["issues"]),
            json.dumps(analysis["statistics"]),
        )
        with self.connect() as connection:
            cursor = connection.execute(query, values)
            analysis_id = cursor.lastrowid
        return self.get(analysis_id)

    def get(self, analysis_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def list(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._to_dict(row) for row in rows]

    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["issues"] = json.loads(result.pop("issues_json"))
        result["statistics"] = json.loads(result.pop("statistics_json"))
        return result
