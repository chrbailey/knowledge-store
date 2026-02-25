"""knowledge-store database access layer — thin wrapper over SQLite + FTS5."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .init_db import get_db_path, init_db


def compute_insight_hash(text: str) -> str:
    """Normalize and hash insight text. Must match knowledge-sync-hook.py."""
    normalized = ' '.join(text.lower().strip().split())[:100]
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


class KnowledgeDB:
    """Single connection wrapper for the knowledge store."""

    def __init__(self, db_path: Union[str, Path, None] = None):
        if db_path is None:
            db_path = get_db_path()
        self._path = Path(db_path)
        self._conn = None  # type: Optional[sqlite3.Connection]

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._path.exists():
                self._conn = init_db(self._path)
            else:
                self._conn = sqlite3.connect(str(self._path))
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Insight CRUD --

    def upsert_insight(
        self,
        text: str,
        tier: int = 3,
        source: str = "auto",
        confidence: float = 0.5,
        session_id: Optional[str] = None,
        project: Optional[str] = None,
        is_personal: bool = False,
        upvotes: int = 0,
        downvotes: int = 0,
        entities: Optional[str] = None,
        relationships: Optional[str] = None,
        relates_to: Optional[str] = None,
        suggested_symbol: Optional[str] = None,
        action: str = "ADD",
        action_target: Optional[str] = None,
        action_reason: Optional[str] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        expired_at: Optional[str] = None,
        drift_eligible: int = 0,
        domain: Optional[str] = None,
        captured_at: Optional[str] = None,
        insight_hash: Optional[str] = None,
    ) -> Optional[int]:
        """Insert insight if hash doesn't exist. Returns rowid or None if duplicate."""
        if insight_hash is None:
            insight_hash = compute_insight_hash(text)
        if captured_at is None:
            captured_at = datetime.utcnow().isoformat()
        if entities is None:
            entities = "[]"
        if relationships is None:
            relationships = "[]"
        if relates_to is None:
            relates_to = "[]"

        try:
            self.execute(
                """INSERT OR IGNORE INTO insights
                   (hash, text, tier, source, confidence, session_id, project,
                    is_personal, upvotes, downvotes, entities, relationships,
                    relates_to, suggested_symbol, action, action_target,
                    action_reason, valid_from, valid_to, expired_at,
                    drift_eligible, domain, captured_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (insight_hash, text, tier, source, confidence, session_id, project,
                 1 if is_personal else 0, upvotes, downvotes, entities, relationships,
                 relates_to, suggested_symbol, action, action_target,
                 action_reason, valid_from, valid_to, expired_at,
                 drift_eligible, domain, captured_at),
            )
            if self.conn.total_changes > 0:
                rowid = self.execute("SELECT last_insert_rowid()").fetchone()[0]
                # Sync to FTS5
                self.execute(
                    """INSERT INTO insights_fts(rowid, text, project, source, entities)
                       VALUES (?, ?, ?, ?, ?)""",
                    (rowid, text, project or "", source, entities),
                )
            self.commit()
            # Check if insert happened
            row = self.fetchone(
                "SELECT id FROM insights WHERE hash = ?", (insight_hash,)
            )
            return row["id"] if row else None
        except sqlite3.IntegrityError:
            return None

    def get_insight(self, ref: str) -> Optional[dict]:
        """Get insight by hash or ID."""
        if ref.isdigit():
            row = self.fetchone("SELECT * FROM insights WHERE id = ?", (int(ref),))
        else:
            row = self.fetchone("SELECT * FROM insights WHERE hash = ?", (ref,))
        return dict(row) if row else None

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip FTS5 operators and special chars to prevent query syntax errors."""
        if not query or not query.strip():
            return ""
        sanitized = re.sub(r'["\(\)\*\^\{\}]', " ", query)
        sanitized = re.sub(r'\b(AND|OR|NOT|NEAR)\b', " ", sanitized)
        sanitized = re.sub(r'\s+', " ", sanitized).strip()
        return sanitized

    def search(
        self,
        query: str,
        limit: int = 10,
        tier_max: int = 3,
        project: Optional[str] = None,
    ) -> List[dict]:
        """FTS5 search with vote-weighted ranking."""
        sanitized = self._sanitize_fts_query(query)
        if not sanitized:
            return []

        # Build WHERE clause
        where_parts = ["insights_fts MATCH ?"]
        params = [sanitized]  # type: List[Any]

        if project:
            where_parts.append("i.project = ?")
            params.append(project)

        where_parts.append("i.tier <= ?")
        params.append(tier_max)

        params.append(limit)

        where_clause = " AND ".join(where_parts)

        rows = self.fetchall(
            f"""SELECT i.*, insights_fts.rank AS fts_rank
                FROM insights_fts
                JOIN insights i ON i.id = insights_fts.rowid
                WHERE {where_clause}
                ORDER BY (
                    insights_fts.rank
                    * CASE i.tier
                        WHEN 0 THEN 2.0
                        WHEN 1 THEN 1.5
                        WHEN 2 THEN 1.0
                        ELSE 0.75
                      END
                    * (1.0 + 0.1 * (i.upvotes - i.downvotes))
                )
                LIMIT ?""",
            tuple(params),
        )
        return [dict(r) for r in rows]

    def vote(self, insight_ref: str, vote_type: str, session_id: Optional[str] = None) -> dict:
        """Record a vote and update insight counters. Returns updated insight."""
        if vote_type not in ("up", "down"):
            raise ValueError("vote_type must be 'up' or 'down'")

        # Resolve ref to hash
        insight = self.get_insight(insight_ref)
        if not insight:
            raise ValueError(f"Insight not found: {insight_ref}")

        insight_hash = insight["hash"]

        self.execute(
            "INSERT INTO votes (insight_hash, vote_type, session_id) VALUES (?, ?, ?)",
            (insight_hash, vote_type, session_id),
        )

        col = "upvotes" if vote_type == "up" else "downvotes"
        self.execute(
            f"UPDATE insights SET {col} = {col} + 1, updated_at = datetime('now') WHERE hash = ?",
            (insight_hash,),
        )
        self.commit()

        return self.get_insight(insight_ref)

    def get_top_insights(self, limit: int = 10, days: int = 7) -> List[dict]:
        """Get top insights ranked by tier weight + net votes, for SessionStart injection."""
        rows = self.fetchall(
            """SELECT *,
                      CASE tier
                          WHEN 0 THEN 4.0
                          WHEN 1 THEN 3.0
                          WHEN 2 THEN 2.0
                          ELSE 1.0
                      END AS tier_weight
               FROM insights
               WHERE captured_at >= datetime('now', ? || ' days')
                 AND expired_at IS NULL
               ORDER BY (tier_weight + (upvotes - downvotes)) DESC,
                        captured_at DESC
               LIMIT ?""",
            (str(-days), limit),
        )
        return [dict(r) for r in rows]

    def pin(self, text: str, project: Optional[str] = None) -> dict:
        """Pin text as a tier-0, high-confidence insight."""
        insight_hash = compute_insight_hash(text)
        self.upsert_insight(
            text=text,
            tier=0,
            source="user_explicit",
            confidence=0.95,
            project=project,
            insight_hash=insight_hash,
        )
        return self.get_insight(insight_hash)

    def stats(self) -> dict:
        """Aggregate stats: total, by tier/source/project, top voted."""
        total = self.fetchone("SELECT COUNT(*) AS c FROM insights")["c"]

        tier_rows = self.fetchall(
            "SELECT tier, COUNT(*) AS c FROM insights GROUP BY tier ORDER BY tier"
        )
        by_tier = {row["tier"]: row["c"] for row in tier_rows}

        source_rows = self.fetchall(
            "SELECT source, COUNT(*) AS c FROM insights GROUP BY source ORDER BY c DESC LIMIT 10"
        )
        by_source = {row["source"]: row["c"] for row in source_rows}

        project_rows = self.fetchall(
            "SELECT project, COUNT(*) AS c FROM insights WHERE project IS NOT NULL GROUP BY project ORDER BY c DESC LIMIT 10"
        )
        by_project = {row["project"]: row["c"] for row in project_rows}

        top_voted = self.fetchall(
            """SELECT hash, text, tier, upvotes, downvotes,
                      (upvotes - downvotes) AS net_votes
               FROM insights
               WHERE (upvotes + downvotes) > 0
               ORDER BY net_votes DESC
               LIMIT 5"""
        )
        top_voted_list = [dict(r) for r in top_voted]

        vote_total = self.fetchone("SELECT COUNT(*) AS c FROM votes")["c"]

        return {
            "total_insights": total,
            "total_votes": vote_total,
            "by_tier": by_tier,
            "by_source": by_source,
            "by_project": by_project,
            "top_voted": top_voted_list,
        }
