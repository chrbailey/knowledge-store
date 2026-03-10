"""knowledge-store database access layer — thin wrapper over SQLite + FTS5."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .init_db import get_db_path, init_db, run_migrations


def compute_insight_hash(text: str) -> str:
    """Normalize and hash insight text. Must match knowledge-sync-hook.py."""
    normalized = ' '.join(text.lower().strip().split())[:100]
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]


# Shared ranking expression for consistent scoring across all query methods.
# Prefix is substituted at call site (e.g., "i." for JOINs, "" for direct queries).
_RANKING_SQL = """(
    CASE {pfx}tier
        WHEN 0 THEN 4.0
        WHEN 1 THEN 3.0
        WHEN 2 THEN 2.0
        ELSE 1.0
    END
    * (1.0 + 0.1 * ({pfx}upvotes - {pfx}downvotes)
       + 0.05 * {pfx}corroboration_count)
    * (1.0 / (1.0 + CAST(
        (julianday('now') - julianday({pfx}captured_at)) AS REAL
    ) * 0.02))
    * (0.5 + {pfx}salience)
)"""


def ranking_sql(prefix: str = "") -> str:
    """Return the ranking expression with an optional table prefix."""
    return _RANKING_SQL.format(pfx=prefix)


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
                run_migrations(self._conn)
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
        salience: float = 0.5,
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
            cursor = self.execute(
                """INSERT OR IGNORE INTO insights
                   (hash, text, tier, source, confidence, session_id, project,
                    is_personal, upvotes, downvotes, entities, relationships,
                    relates_to, suggested_symbol, action, action_target,
                    action_reason, valid_from, valid_to, expired_at,
                    drift_eligible, domain, captured_at, salience)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (insight_hash, text, tier, source, confidence, session_id, project,
                 1 if is_personal else 0, upvotes, downvotes, entities, relationships,
                 relates_to, suggested_symbol, action, action_target,
                 action_reason, valid_from, valid_to, expired_at,
                 drift_eligible, domain, captured_at, salience),
            )
            if cursor.rowcount > 0:
                rowid = cursor.lastrowid
                # Sync to FTS5
                self.execute(
                    """INSERT INTO insights_fts(rowid, text, project, source, entities)
                       VALUES (?, ?, ?, ?, ?)""",
                    (rowid, text, project or "", source, entities),
                )
                self.commit()
                return rowid
            self.commit()
            return None  # duplicate
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
                  AND i.expired_at IS NULL
                ORDER BY insights_fts.rank * {ranking_sql("i.")}
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
        """Get top insights with unified ranking: tier * votes * recency * salience."""
        rows = self.fetchall(
            f"""SELECT *
               FROM insights
               WHERE captured_at >= datetime('now', ? || ' days')
                 AND expired_at IS NULL
               ORDER BY {ranking_sql()} DESC
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

    def get_project_insights(
        self,
        project: str,
        limit: int = 10,
        tier_max: int = 3,
        days: int = 30,
    ) -> List[dict]:
        """Get insights for a specific project with unified ranking.

        Uses the project column directly (NOT FTS5 keyword search).
        This is the correct way to get project-scoped context at session start.
        """
        rows = self.fetchall(
            f"""SELECT *
               FROM insights
               WHERE project = ?
                 AND tier <= ?
                 AND captured_at >= datetime('now', ? || ' days')
                 AND expired_at IS NULL
               ORDER BY {ranking_sql()} DESC
               LIMIT ?""",
            (project, tier_max, str(-days), limit),
        )
        return [dict(r) for r in rows]

    def increment_corroboration(self, insight_hash: str) -> bool:
        """Increment corroboration count for an existing insight.

        Called when the stop hook re-extracts the same insight from a new session.
        Higher corroboration = higher trust signal. Returns True if row was updated.
        """
        cursor = self.execute(
            """UPDATE insights
               SET corroboration_count = corroboration_count + 1,
                   updated_at = datetime('now')
               WHERE hash = ?""",
            (insight_hash,),
        )
        self.commit()
        return cursor.rowcount > 0

    def expire(self, insight_hash: str) -> bool:
        """Soft-expire a single insight. Returns True if found and expired."""
        cursor = self.execute(
            """UPDATE insights SET expired_at = datetime('now'),
                   updated_at = datetime('now')
               WHERE hash = ? AND expired_at IS NULL""",
            (insight_hash,),
        )
        self.commit()
        return cursor.rowcount > 0

    def promote(self, insight_hash: str, new_tier: int = 1,
                confidence_boost: float = 0.2) -> bool:
        """Promote an insight to a higher tier and re-index in FTS5."""
        cursor = self.execute(
            """UPDATE insights SET tier = ?,
                   confidence = MIN(1.0, confidence + ?),
                   updated_at = datetime('now')
               WHERE hash = ?""",
            (new_tier, confidence_boost, insight_hash),
        )
        if cursor.rowcount == 0:
            return False
        # Re-index in FTS5
        row = self.fetchone(
            "SELECT id FROM insights WHERE hash = ?", (insight_hash,)
        )
        if row:
            self.execute(
                "INSERT INTO insights_fts(insights_fts, rowid) VALUES('delete', ?)",
                (row["id"],),
            )
            self.execute(
                "INSERT INTO insights_fts(rowid, text, project, source, entities) "
                "SELECT id, text, project, source, entities FROM insights WHERE id = ?",
                (row["id"],),
            )
        self.commit()
        return True

    def transfer_signals(self, to_hash: str, upvotes: int = 0,
                         downvotes: int = 0, corroboration: int = 0) -> None:
        """Transfer aggregate signals (votes, corroboration) to a surviving insight."""
        if upvotes or downvotes or corroboration:
            self.execute(
                """UPDATE insights SET upvotes = upvotes + ?,
                       downvotes = downvotes + ?,
                       corroboration_count = corroboration_count + ?,
                       updated_at = datetime('now')
                   WHERE hash = ?""",
                (upvotes, downvotes, corroboration, to_hash),
            )
            self.commit()

    def prune_stale(self, max_age_days: int = 90, min_tier: int = 3) -> int:
        """Soft-expire insights that are old, unvoted, uncorroborated, and low-tier.

        Criteria: older than max_age_days AND zero net votes AND
        corroboration_count = 0 AND tier >= min_tier AND not already expired.

        Returns count of pruned insights.
        """
        cursor = self.execute(
            """UPDATE insights
               SET expired_at = datetime('now'),
                   updated_at = datetime('now')
               WHERE captured_at < datetime('now', ? || ' days')
                 AND (upvotes - downvotes) <= 0
                 AND corroboration_count = 0
                 AND tier >= ?
                 AND expired_at IS NULL""",
            (str(-max_age_days), min_tier),
        )
        pruned = cursor.rowcount
        self.commit()
        return pruned

    def get_investigation_prompts(
        self,
        project: Optional[str] = None,
        limit: int = 2,
    ) -> List[str]:
        """Generate investigation prompts from knowledge gaps.

        Returns human-readable prompts about:
        1. Topics with low-confidence recent insights
        2. Projects with stale insights (30+ days, no recent activity)
        3. Highly corroborated topics that may have evolved

        Used by SessionStart to inject alongside factual context.
        """
        prompts = []  # type: List[str]

        # 1. Low-confidence recent insights (last 14 days, confidence < 0.5)
        low_conf_rows = self.fetchall(
            """SELECT text, project, confidence, captured_at
               FROM insights
               WHERE confidence < 0.5
                 AND captured_at >= datetime('now', '-14 days')
                 AND expired_at IS NULL
               ORDER BY captured_at DESC
               LIMIT 3""",
        )
        for row in low_conf_rows:
            d = dict(row)
            text_preview = d["text"][:80]
            proj = d.get("project") or "general"
            prompts.append(
                f"[investigate] Low-confidence insight ({proj}): \"{text_preview}...\" -- can you verify?"
            )
            if len(prompts) >= limit:
                return prompts

        # 2. Stale active projects -- projects where most recent insight is 30+ days old
        stale_rows = self.fetchall(
            """SELECT project, MAX(captured_at) AS last_insight, COUNT(*) AS insight_count
               FROM insights
               WHERE expired_at IS NULL
                 AND project IS NOT NULL
               GROUP BY project
               HAVING MAX(captured_at) < datetime('now', '-30 days')
                  AND COUNT(*) >= 3
               ORDER BY insight_count DESC
               LIMIT 3""",
        )
        for row in stale_rows:
            d = dict(row)
            proj = d["project"]
            count = d["insight_count"]
            if project and proj != project:
                continue  # Only surface stale prompts for current project
            prompts.append(
                f"[investigate] {count} insights about '{proj}' haven't been updated in 30+ days -- still accurate?"
            )
            if len(prompts) >= limit:
                return prompts

        # 3. Highly corroborated insights that may have evolved
        evolving_rows = self.fetchall(
            """SELECT text, project, corroboration_count, captured_at
               FROM insights
               WHERE corroboration_count >= 3
                 AND captured_at < datetime('now', '-30 days')
                 AND expired_at IS NULL
               ORDER BY corroboration_count DESC
               LIMIT 3""",
        )
        for row in evolving_rows:
            d = dict(row)
            text_preview = d["text"][:80]
            prompts.append(
                f"[investigate] Frequently rediscovered ({d['corroboration_count']}x): \"{text_preview}...\" -- has this changed?"
            )
            if len(prompts) >= limit:
                return prompts

        return prompts[:limit]

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
