#!/usr/bin/env python3
"""
Deduplicate knowledge.db insights.

For each cluster of near-duplicates (same first 60 chars lowered):
- Keep the one with highest quality (tier, votes, corroboration)
- Expire the rest
- Transfer any votes/corroboration to the survivor

Also handles:
- Expired project references (Pinecone insights when Pinecone is retired)
- Insights too short to be useful after trimming
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB  # noqa: E402


def quality_key(row: dict) -> tuple:
    """Sort key: lower tier is better, more votes, more corroboration, longer text."""
    return (
        -row.get("tier", 3),                    # lower tier = better (negate for sort)
        row.get("upvotes", 0) - row.get("downvotes", 0),  # net votes
        row.get("corroboration_count", 0),
        len(row.get("text", "")),                # longer = more context
    )


def find_duplicate_clusters(db: KnowledgeDB) -> list:
    """Find clusters of near-duplicate insights."""
    rows = db.fetchall(
        """SELECT id, hash, text, tier, upvotes, downvotes, corroboration_count,
                  confidence, salience, project, source, captured_at
           FROM insights WHERE expired_at IS NULL
           ORDER BY LOWER(SUBSTR(text, 1, 60))"""
    )

    clusters = {}
    for row in rows:
        r = dict(row)
        key = r["text"][:60].lower().strip()
        clusters.setdefault(key, []).append(r)

    # Only return clusters with duplicates
    return {k: v for k, v in clusters.items() if len(v) > 1}


def find_retired_project_insights(db: KnowledgeDB) -> list:
    """Find insights about retired technologies."""
    retired_terms = ["pinecone", "nanomemory", "aether"]
    results = []
    for term in retired_terms:
        rows = db.fetchall(
            "SELECT id, hash, text, tier, project FROM insights "
            "WHERE expired_at IS NULL AND LOWER(text) LIKE ?",
            (f"%{term}%",)
        )
        results.extend(dict(r) for r in rows)
    return results


def main():
    db = KnowledgeDB()
    dry_run = "--dry-run" in sys.argv

    # Phase 1: Deduplicate clusters
    clusters = find_duplicate_clusters(db)
    total_dupes = 0
    total_expired = 0

    print(f"Found {len(clusters)} duplicate clusters\n")

    for key, members in clusters.items():
        members.sort(key=quality_key, reverse=True)
        survivor = members[0]
        dupes = members[1:]

        # Transfer votes and corroboration to survivor
        extra_up = sum(d.get("upvotes", 0) for d in dupes)
        extra_down = sum(d.get("downvotes", 0) for d in dupes)
        extra_corr = sum(d.get("corroboration_count", 0) for d in dupes)

        print(f"Cluster: \"{key[:50]}...\"")
        print(f"  Keep:   [{survivor['hash']}] tier={survivor['tier']} "
              f"votes={survivor.get('upvotes',0)}")
        for d in dupes:
            print(f"  Expire: [{d['hash']}] tier={d['tier']}")

        if not dry_run:
            # Transfer aggregate signals
            if extra_up or extra_down or extra_corr:
                db.conn.execute(
                    "UPDATE insights SET upvotes = upvotes + ?, downvotes = downvotes + ?, "
                    "corroboration_count = corroboration_count + ?, updated_at = datetime('now') "
                    "WHERE hash = ?",
                    (extra_up, extra_down, extra_corr, survivor["hash"])
                )

            # Expire duplicates
            for d in dupes:
                db.conn.execute(
                    "UPDATE insights SET expired_at = datetime('now'), "
                    "updated_at = datetime('now') WHERE hash = ?",
                    (d["hash"],)
                )

            db.conn.commit()

        total_dupes += len(members)
        total_expired += len(dupes)

    # Phase 2: Expire retired project insights
    retired = find_retired_project_insights(db)
    retired_expired = 0

    if retired:
        print(f"\nRetired technology insights: {len(retired)}")
        for r in retired:
            # Don't expire if it's a meta-lesson (tier 0-1)
            if r.get("tier", 3) <= 1:
                print(f"  Keep (high tier): [{r['hash']}] {r['text'][:60]}...")
                continue
            print(f"  Expire: [{r['hash']}] {r['text'][:60]}...")
            if not dry_run:
                db.conn.execute(
                    "UPDATE insights SET expired_at = datetime('now'), "
                    "updated_at = datetime('now') WHERE hash = ?",
                    (r["hash"],)
                )
                retired_expired += 1

        if not dry_run:
            db.conn.commit()

    # Summary
    print(f"\n{'DRY RUN — ' if dry_run else ''}Summary:")
    print(f"  Duplicate clusters: {len(clusters)}")
    print(f"  Total duplicates:   {total_dupes}")
    print(f"  Expired (dedup):    {total_expired}")
    print(f"  Expired (retired):  {retired_expired}")

    # Post-cleanup count
    if not dry_run:
        count = db.fetchall("SELECT COUNT(*) as cnt FROM insights WHERE expired_at IS NULL")[0]["cnt"]
        print(f"  Active insights:    {count}")

    db.close()


if __name__ == "__main__":
    main()
