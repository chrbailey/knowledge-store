"""knowledge-store database initialization — SQLite + FTS5."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS insights (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hash          TEXT UNIQUE NOT NULL,
    text          TEXT NOT NULL,
    tier          INTEGER NOT NULL DEFAULT 3,
    source        TEXT NOT NULL DEFAULT 'auto',
    confidence    REAL NOT NULL DEFAULT 0.5,
    session_id    TEXT,
    project       TEXT,
    is_personal   INTEGER DEFAULT 0,
    upvotes       INTEGER DEFAULT 0,
    downvotes     INTEGER DEFAULT 0,
    entities      TEXT DEFAULT '[]',
    relationships TEXT DEFAULT '[]',
    relates_to    TEXT DEFAULT '[]',
    suggested_symbol TEXT,
    action        TEXT DEFAULT 'ADD',
    action_target TEXT,
    action_reason TEXT,
    valid_from    TEXT,
    valid_to      TEXT,
    expired_at    TEXT,
    drift_eligible INTEGER DEFAULT 0,
    domain        TEXT,
    captured_at   TEXT NOT NULL DEFAULT (datetime('now')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
    text, project, source, entities,
    content=insights,
    content_rowid=id,
    tokenize='porter'
);

CREATE TABLE IF NOT EXISTS votes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_hash  TEXT NOT NULL REFERENCES insights(hash),
    vote_type     TEXT NOT NULL CHECK (vote_type IN ('up', 'down')),
    session_id    TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_insights_hash ON insights(hash);
CREATE INDEX IF NOT EXISTS idx_insights_tier ON insights(tier);
CREATE INDEX IF NOT EXISTS idx_insights_captured ON insights(captured_at);
CREATE INDEX IF NOT EXISTS idx_insights_project ON insights(project);
CREATE INDEX IF NOT EXISTS idx_votes_hash ON votes(insight_hash);
"""

DEFAULT_DB_PATH = Path("/Volumes/OWC drive/Knowledge/knowledge.db")

MIGRATIONS = [
    # Migration 1: Add salience column (Phase 3)
    (
        "salience_column",
        "ALTER TABLE insights ADD COLUMN salience REAL NOT NULL DEFAULT 0.5",
    ),
    # Migration 2: Add corroboration count (Phase 4)
    (
        "corroboration_column",
        "ALTER TABLE insights ADD COLUMN corroboration_count INTEGER NOT NULL DEFAULT 0",
    ),
]


def run_migrations(conn: sqlite3.Connection) -> None:
    """Run pending schema migrations. Idempotent."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()

    for name, sql in MIGRATIONS:
        existing = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?", (name,)
        ).fetchone()
        if existing:
            continue
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO schema_migrations (name) VALUES (?)", (name,)
            )
            conn.commit()
        except sqlite3.OperationalError:
            # Column already exists (manual migration), just record it
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (name) VALUES (?)", (name,)
            )
            conn.commit()


def get_db_path(data_dir: Optional[str] = None) -> Path:
    if data_dir:
        return Path(data_dir) / "knowledge.db"
    env_path = os.environ.get("KNOWLEDGE_DB")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_PATH


def init_db(db_path: Union[str, Path] = None) -> sqlite3.Connection:
    """Initialize the knowledge database. Idempotent."""
    if db_path is None:
        db_path = get_db_path()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    run_migrations(conn)
    return conn


def get_connection(db_path: Union[str, Path] = None) -> sqlite3.Connection:
    """Get a connection to an existing DB (initializes if needed)."""
    if db_path is None:
        db_path = get_db_path()
    db_path = Path(db_path)
    if not db_path.exists():
        return init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    db_path = Path(path) if path else get_db_path()
    conn = init_db(db_path)

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('insights_fts_')]
    print(f"knowledge database initialized: {db_path}")
    print(f"Tables created: {len(tables)}")
    for t in tables:
        print(f"  - {t}")
    conn.close()
