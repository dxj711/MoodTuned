"""SQLite persistence for MoodTune AI."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class MoodDatabase:
    """Stores previous image moods, recommendations, and lightweight embeddings."""

    def __init__(self, db_path: str | Path = "moodtune_history.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def save_search(
        self,
        image_hash: str,
        image_name: str,
        thumbnail_b64: str,
        features: dict[str, Any],
        analysis: dict[str, Any],
        recommendations: list[dict[str, Any]],
        language: str,
        genre_pref: str,
        style: str,
        embedding: list[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO searches (
                    created_at, image_hash, image_name, thumbnail_b64, caption,
                    main_emotion, mood_tags_json, vibe_description, recommended_genre,
                    language, genre_pref, style, recommendations_json,
                    features_json, analysis_json, embedding_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    image_hash,
                    image_name,
                    thumbnail_b64,
                    features.get("caption", ""),
                    analysis.get("main_emotion", ""),
                    json.dumps(analysis.get("mood_tags", [])),
                    analysis.get("vibe_description", ""),
                    analysis.get("recommended_genre", ""),
                    language,
                    genre_pref,
                    style,
                    json.dumps(recommendations),
                    json.dumps(features),
                    json.dumps(analysis),
                    json.dumps(embedding),
                ),
            )

    def get_history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM searches ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_similar_searches(self, embedding: list[float], limit: int = 3) -> list[dict[str, Any]]:
        history = self.get_history(limit=80)
        scored = []
        for row in history:
            past_embedding = row.get("embedding", [])
            if not past_embedding:
                continue
            similarity = self._cosine_similarity(embedding, past_embedding)
            if similarity >= 0.72:
                row["similarity"] = similarity
                scored.append(row)
        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM searches")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    image_hash TEXT NOT NULL,
                    image_name TEXT,
                    thumbnail_b64 TEXT,
                    caption TEXT,
                    main_emotion TEXT,
                    mood_tags_json TEXT,
                    vibe_description TEXT,
                    recommended_genre TEXT,
                    language TEXT,
                    genre_pref TEXT,
                    style TEXT,
                    recommendations_json TEXT,
                    features_json TEXT,
                    analysis_json TEXT,
                    embedding_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_searches_created_at ON searches(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_searches_hash ON searches(image_hash)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["mood_tags"] = self._json_load(data.pop("mood_tags_json", "[]"), [])
        data["recommendations"] = self._json_load(data.pop("recommendations_json", "[]"), [])
        data["features"] = self._json_load(data.pop("features_json", "{}"), {})
        data["analysis"] = self._json_load(data.pop("analysis_json", "{}"), {})
        data["embedding"] = self._json_load(data.pop("embedding_json", "[]"), [])
        return data

    def _json_load(self, text: str | None, fallback: Any) -> Any:
        try:
            return json.loads(text or "")
        except Exception:
            return fallback

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        size = min(len(a), len(b))
        a = a[:size]
        b = b[:size]
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)
