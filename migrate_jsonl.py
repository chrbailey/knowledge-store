#!/usr/bin/env python3
"""One-time migration: captured_insights.jsonl -> knowledge.db SQLite.

Idempotent — safe to run multiple times (INSERT OR IGNORE on hash).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add self to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_lib.db import KnowledgeDB, compute_insight_hash

JSONL_PATH = Path("/Volumes/OWC drive/Knowledge/captured_insights.jsonl")
DB_PATH = Path("/Volumes/OWC drive/Knowledge/knowledge.db")


def migrate(jsonl_path: Path = JSONL_PATH, db_path: Path = DB_PATH) -> dict:
    """Read JSONL, insert into SQLite. Returns stats."""
    if not jsonl_path.exists():
        print(f"JSONL not found: {jsonl_path}")
        return {"error": "JSONL not found"}

    db = KnowledgeDB(db_path=db_path)
    stats = {"total": 0, "inserted": 0, "skipped": 0, "errors": 0}

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["errors"] += 1
                print(f"  Line {line_num}: JSON parse error, skipping")
                continue

            text = record.get("text", "")
            if not text:
                stats["skipped"] += 1
                continue

            insight_hash = compute_insight_hash(text)

            # Extract fields, with defaults for older records
            entities = record.get("entities", [])
            if isinstance(entities, list):
                entities = json.dumps(entities)
            relationships = record.get("relationships", [])
            if isinstance(relationships, list):
                relationships = json.dumps(relationships)
            relates_to = record.get("relates_to", [])
            if isinstance(relates_to, list):
                relates_to = json.dumps(relates_to)

            # Check if already exists before insert
            existing = db.get_insight(insight_hash)
            if existing:
                stats["skipped"] += 1
                continue

            db.upsert_insight(
                text=text,
                tier=record.get("tier", 3),
                source=record.get("source", "auto"),
                confidence=record.get("confidence", 0.5),
                session_id=record.get("session_id"),
                project=record.get("project"),
                is_personal=record.get("is_personal", False),
                upvotes=record.get("upvotes", 0),
                downvotes=record.get("downvotes", 0),
                entities=entities,
                relationships=relationships,
                relates_to=relates_to,
                suggested_symbol=record.get("suggested_symbol"),
                action=record.get("action", "ADD"),
                action_target=record.get("action_target"),
                action_reason=record.get("action_reason"),
                valid_from=record.get("valid_from"),
                valid_to=record.get("valid_to"),
                expired_at=record.get("expired_at"),
                drift_eligible=record.get("drift_eligible", 0),
                domain=record.get("domain"),
                captured_at=record.get("captured_at", record.get("timestamp")),
                insight_hash=insight_hash,
            )
            stats["inserted"] += 1

    db.close()
    return stats


def main():
    print(f"Migrating: {JSONL_PATH}")
    print(f"Target DB: {DB_PATH}")
    print()

    stats = migrate()

    print(f"Total records:  {stats.get('total', 0)}")
    print(f"Inserted:       {stats.get('inserted', 0)}")
    print(f"Skipped (dedup): {stats.get('skipped', 0)}")
    print(f"Errors:         {stats.get('errors', 0)}")


if __name__ == "__main__":
    main()
