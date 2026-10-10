"""Session manager — project & run storage backed by filesystem + SQLite."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path("./endforecast_sessions")


class SessionManager:
    """Manages projects and pipeline runs.

    Storage layout:
        sessions/
        ├── sessions.db          # SQLite index (fast listing/queries)
        └── proj_<uuid>/
            ├── meta.json        # Project config
            ├── data.csv         # Uploaded data file
            └── runs/
                └── run_<uuid>/
                    ├── result.json   # RunResult as JSON
                    ├── predictor.py  # Exported code
                    └── logs/
    """

    def __init__(self, base_dir: str | Path = DEFAULT_SESSIONS_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.base_dir / "sessions.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    status TEXT DEFAULT 'idle',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    n_rows INTEGER,
                    n_cols INTEGER,
                    task_type TEXT,
                    latest_holdout_score REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    elapsed_seconds REAL,
                    summary TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL UNIQUE,
                    prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used TEXT
                )
            """)
            conn.commit()

    # ── Project CRUD ──────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        target_col: str = "y",
        time_col: Optional[str] = None,
        id_col: Optional[str] = None,
        budget: str = "auto",
        data_bytes: Optional[bytes] = None,
        data_filename: str = "data.csv",
    ) -> dict:
        proj_id = f"proj_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        proj_dir = self.base_dir / proj_id
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "runs").mkdir(exist_ok=True)

        # Save uploaded data
        data_path = proj_dir / data_filename
        if data_bytes:
            suffix = Path(data_filename).suffix or ".csv"
            data_path = proj_dir / f"data{suffix}"
            data_path.write_bytes(data_bytes)

        # Save metadata
        meta = {
            "id": proj_id, "name": name, "description": description,
            "target_col": target_col, "time_col": time_col,
            "id_col": id_col, "budget": budget,
            "data_path": str(data_path),
            "created_at": now, "updated_at": now,
        }
        (proj_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO projects (id, name, description, status, created_at, updated_at) VALUES (?, ?, ?, 'idle', ?, ?)",
                (proj_id, name, description, now, now),
            )
            conn.commit()

        return meta

    def list_projects(self) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_project(self, proj_id: str) -> Optional[dict]:
        proj_dir = self.base_dir / proj_id
        meta_path = proj_dir / "meta.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        # Merge with DB fields
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?", (proj_id,)
            ).fetchone()
        if row:
            db_data = dict(row)
            meta.update({k: v for k, v in db_data.items() if k not in meta})
        return meta

    def update_project(self, proj_id: str, **fields) -> None:
        meta = self.get_project(proj_id)
        if not meta:
            return
        meta.update(fields)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        proj_dir = self.base_dir / proj_id
        (proj_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values())
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                f"UPDATE projects SET {cols}, updated_at=? WHERE id=?",
                vals + [meta["updated_at"], proj_id],
            )
            conn.commit()

    def delete_project(self, proj_id: str) -> None:
        proj_dir = self.base_dir / proj_id
        if proj_dir.exists():
            import shutil
            shutil.rmtree(proj_dir)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM runs WHERE project_id = ?", (proj_id,))
            conn.execute("DELETE FROM projects WHERE id = ?", (proj_id,))
            conn.commit()

    def project_data_path(self, proj_id: str) -> Optional[Path]:
        meta = self.get_project(proj_id)
        if not meta:
            return None
        p = Path(meta["data_path"])
        return p if p.exists() else None

    # ── Run management ────────────────────────────────────────

    def create_run(self, proj_id: str) -> str:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        run_dir = self.base_dir / proj_id / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO runs (id, project_id, status, started_at) VALUES (?, ?, 'running', ?)",
                (run_id, proj_id, now),
            )
            conn.execute(
                "UPDATE projects SET status='running', updated_at=? WHERE id=?",
                (now, proj_id),
            )
            conn.commit()
        return run_id

    def complete_run(self, proj_id: str, run_id: str, result_dict: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        run_dir = self.base_dir / proj_id / "runs" / run_id
        (run_dir / "result.json").write_text(json.dumps(result_dict, default=str, indent=2))

        elapsed = result_dict.get("elapsed_seconds", 0)
        summary = result_dict.get("summary", "")
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE runs SET status='completed', completed_at=?, elapsed_seconds=?, summary=? WHERE id=?",
                (now, elapsed, summary, run_id),
            )
            conn.execute(
                "UPDATE projects SET status='completed', updated_at=?, latest_holdout_score=? WHERE id=?",
                (now, result_dict.get("holdout_score"), proj_id),
            )
            conn.commit()

    def fail_run(self, proj_id: str, run_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE runs SET status='failed', completed_at=? WHERE id=?",
                (now, run_id),
            )
            conn.execute(
                "UPDATE projects SET status='failed', updated_at=? WHERE id=?",
                (now, proj_id),
            )
            conn.commit()

    def list_runs(self, proj_id: str) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs WHERE project_id = ? ORDER BY started_at DESC",
                (proj_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, proj_id: str, run_id: str) -> Optional[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result_path = self.base_dir / proj_id / "runs" / run_id / "result.json"
        if result_path.exists():
            result["result"] = json.loads(result_path.read_text())
        return result

    def run_dir(self, proj_id: str, run_id: str) -> Path:
        d = self.base_dir / proj_id / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── API keys ──────────────────────────────────────────────

    def create_api_key(self, name: str) -> tuple[dict, str]:
        from endforecast.web.models.api_key import ApiKey
        ak, raw = ApiKey.generate(name)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO api_keys (id, name, key_hash, prefix, created_at) VALUES (?, ?, ?, ?, ?)",
                (ak.id, ak.name, ak.key_hash, ak.prefix, ak.created_at),
            )
            conn.commit()
        return {"id": ak.id, "name": ak.name, "prefix": ak.prefix, "created_at": ak.created_at}, raw

    def list_api_keys(self) -> list[dict]:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, name, prefix, created_at, last_used FROM api_keys ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def verify_api_key(self, raw_key: str) -> bool:
        import hashlib
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        with sqlite3.connect(str(self._db_path)) as conn:
            row = conn.execute("SELECT id FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()
        return row is not None

    def delete_api_key(self, key_id: str) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
            conn.commit()