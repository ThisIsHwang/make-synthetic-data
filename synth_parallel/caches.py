from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .io_utils import ensure_parent


class SQLiteKVCache:
    def __init__(self, db_path: str | Path, table_name: str):
        self.db_path = Path(db_path)
        ensure_parent(self.db_path)
        self.table_name = table_name
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                cache_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT value_json FROM {self.table_name} WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.table_name}(cache_key, value_json, created_at) VALUES (?, ?, ?)",
                (key, payload, time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class StageProgressStore:
    """Line-level resume tracker for large stages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_parent(self.db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stage_progress (
                stage_name TEXT NOT NULL,
                item_id TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(stage_name, item_id)
            )
            """
        )
        self._conn.commit()

    def has(self, stage_name: str, item_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM stage_progress WHERE stage_name = ? AND item_id = ?",
                (stage_name, item_id),
            ).fetchone()
        return row is not None

    def add(self, stage_name: str, item_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO stage_progress(stage_name, item_id, updated_at) VALUES (?, ?, ?)",
                (stage_name, item_id, time.time()),
            )
            self._conn.commit()

    def add_many(self, stage_name: str, item_ids: list[str]) -> None:
        if not item_ids:
            return
        rows = [(stage_name, item_id, time.time()) for item_id in item_ids]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO stage_progress(stage_name, item_id, updated_at) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def count_stage(self, stage_name: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM stage_progress WHERE stage_name = ?",
                (stage_name,),
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
