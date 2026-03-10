#!/usr/bin/env python3
"""
Export knowledge.db as training-ready JSONL.

Karpathy's rule: your model is only as good as your dataset.
This script converts the knowledge store into labeled training data.

Each insight becomes a training example with:
  - text: the insight content
  - quality_score: composite of tier, votes, corroboration, salience
  - preference: derived from votes (positive = upvoted, negative = downvoted)
  - provenance: session, project, timestamp, source

Output formats:
  1. Raw JSONL (all fields, for analysis)
  2. Chat JSONL (OpenAI/Anthropic fine-tuning format)
  3. Preference pairs (DPO/RLHF format — high-quality vs low-quality)

Usage:
  python3 knowledge-export.py                    # raw JSONL to stdout
  python3 knowledge-export.py --format chat      # chat format
  python3 knowledge-export.py --format preference # preference pairs
  python3 knowledge-export.py --project promptspeak --min-quality 0.5
  python3 knowledge-export.py --stats             # dataset statistics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB  # noqa: E402


def quality_score(row: dict) -> float:
    """Composite quality score (0.0 - 1.0). This is the reward signal."""
    tier_weight = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2}
    tier_w = tier_weight.get(row.get("tier", 3), 0.2)

    net_votes = row.get("upvotes", 0) - row.get("downvotes", 0)
    vote_signal = min(1.0, max(-1.0, net_votes * 0.2))

    corr = min(1.0, row.get("corroboration_count", 0) * 0.25)
    salience = row.get("salience", 0.5)
    confidence = row.get("confidence", 0.5)

    # Weighted combination
    score = (
        0.30 * tier_w +
        0.20 * ((vote_signal + 1) / 2) +  # normalize -1..1 to 0..1
        0.20 * corr +
        0.15 * salience +
        0.15 * confidence
    )
    return round(min(1.0, max(0.0, score)), 4)


def export_raw(rows: List[dict]):
    """Full insight records with computed quality score."""
    for row in rows:
        record = {
            "text": row["text"],
            "quality_score": quality_score(row),
            "tier": row.get("tier"),
            "confidence": row.get("confidence"),
            "salience": row.get("salience"),
            "upvotes": row.get("upvotes", 0),
            "downvotes": row.get("downvotes", 0),
            "corroboration_count": row.get("corroboration_count", 0),
            "project": row.get("project"),
            "source": row.get("source"),
            "session_id": row.get("session_id"),
            "captured_at": row.get("captured_at"),
            "is_personal": bool(row.get("is_personal")),
        }
        print(json.dumps(record))


def export_chat(rows: List[dict]):
    """Chat fine-tuning format (OpenAI/Anthropic compatible).

    Each insight becomes a system+assistant turn:
    system: "You are a helpful AI assistant working on {project}."
    assistant: "{insight text}"

    The quality_score can be used for filtering or weighting.
    """
    for row in rows:
        q = quality_score(row)
        project = row.get("project") or "general"
        record = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an AI assistant working on the {project} project. "
                               "Share relevant technical insights and patterns.",
                },
                {
                    "role": "assistant",
                    "content": row["text"],
                },
            ],
            "quality_score": q,
            "metadata": {
                "project": project,
                "tier": row.get("tier"),
                "source": row.get("source"),
            },
        }
        print(json.dumps(record))


def export_preference(rows: List[dict]):
    """DPO/RLHF preference pairs: high-quality vs low-quality insights.

    Pairs insights from the same project where one is clearly
    better (tier 0-1 vs tier 3, or upvoted vs downvoted).
    This is how you turn a knowledge store into a preference dataset.
    """
    by_project = {}  # type: dict
    for row in rows:
        project = row.get("project") or "general"
        by_project.setdefault(project, []).append(row)

    for project, project_rows in by_project.items():
        scored = [(quality_score(r), r) for r in project_rows]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Pair top quartile with bottom quartile
        n = len(scored)
        if n < 4:
            continue
        top = scored[:n // 4]
        bottom = scored[-(n // 4):]

        for (q_good, good), (q_bad, bad) in zip(top, bottom):
            if q_good - q_bad < 0.2:
                continue  # not enough quality gap
            record = {
                "prompt": f"Share a technical insight about {project}.",
                "chosen": good["text"],
                "rejected": bad["text"],
                "chosen_quality": q_good,
                "rejected_quality": q_bad,
                "project": project,
            }
            print(json.dumps(record))


def print_stats(rows: List[dict]):
    """Dataset statistics — the Karpathy checklist."""
    total = len(rows)
    if total == 0:
        print("Empty dataset.")
        return

    scores = [quality_score(r) for r in rows]
    avg_q = sum(scores) / len(scores)
    high_q = sum(1 for s in scores if s >= 0.6)
    low_q = sum(1 for s in scores if s < 0.3)

    tiers = {}
    sources = {}
    projects = {}
    for r in rows:
        t = r.get("tier", 3)
        tiers[t] = tiers.get(t, 0) + 1
        s = r.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
        p = r.get("project") or "none"
        projects[p] = projects.get(p, 0) + 1

    total_votes = sum(r.get("upvotes", 0) + r.get("downvotes", 0) for r in rows)
    total_corr = sum(r.get("corroboration_count", 0) for r in rows)
    avg_len = sum(len(r.get("text", "")) for r in rows) / total

    print(f"Dataset Summary")
    print(f"  Total examples:     {total}")
    print(f"  Avg quality score:  {avg_q:.3f}")
    print(f"  High quality (≥0.6): {high_q} ({100*high_q/total:.0f}%)")
    print(f"  Low quality (<0.3):  {low_q} ({100*low_q/total:.0f}%)")
    print(f"  Avg text length:    {avg_len:.0f} chars")
    print(f"  Total votes:        {total_votes}")
    print(f"  Total corroborations: {total_corr}")
    print(f"\nBy tier:")
    for t in sorted(tiers):
        label = {0: "pinned", 1: "discovery", 2: "general", 3: "observation"}.get(t, f"tier-{t}")
        print(f"  {label}: {tiers[t]}")
    print(f"\nBy source:")
    for s, c in sorted(sources.items(), key=lambda x: -x[1])[:5]:
        print(f"  {s}: {c}")
    print(f"\nBy project (top 10):")
    for p, c in sorted(projects.items(), key=lambda x: -x[1])[:10]:
        print(f"  {p}: {c}")

    # Data quality warnings
    print(f"\nData quality signals:")
    if total_votes == 0:
        print("  ⚠ No votes recorded — preference signal missing")
    if avg_q < 0.4:
        print("  ⚠ Low avg quality — consider pruning tier-3 observations")
    if high_q / total > 0.5:
        print("  ✓ Majority high-quality examples")
    voted = sum(1 for r in rows if r.get("upvotes", 0) + r.get("downvotes", 0) > 0)
    print(f"  Voted examples: {voted}/{total} ({100*voted/total:.0f}%)")
    corr_examples = sum(1 for r in rows if r.get("corroboration_count", 0) > 0)
    print(f"  Corroborated: {corr_examples}/{total} ({100*corr_examples/total:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Export knowledge.db as training JSONL")
    parser.add_argument("--format", choices=["raw", "chat", "preference"], default="raw")
    parser.add_argument("--project", help="Filter by project")
    parser.add_argument("--min-quality", type=float, default=0.0, help="Min quality score")
    parser.add_argument("--min-tier", type=int, default=0, help="Min tier (0=pinned)")
    parser.add_argument("--max-tier", type=int, default=3, help="Max tier (3=observation)")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    parser.add_argument("--db", help="Path to knowledge.db", default=None)
    args = parser.parse_args()

    db = KnowledgeDB(db_path=args.db) if args.db else KnowledgeDB()

    # Fetch all active insights
    rows = db.fetchall(
        "SELECT * FROM insights WHERE expired_at IS NULL ORDER BY captured_at DESC"
    )
    rows = [dict(r) for r in rows]
    db.close()

    # Apply filters
    if args.project:
        rows = [r for r in rows if r.get("project") == args.project]
    if args.min_tier > 0 or args.max_tier < 3:
        rows = [r for r in rows if args.min_tier <= r.get("tier", 3) <= args.max_tier]
    if args.min_quality > 0:
        rows = [r for r in rows if quality_score(r) >= args.min_quality]

    if args.stats:
        print_stats(rows)
        return

    exporters = {"raw": export_raw, "chat": export_chat, "preference": export_preference}
    exporters[args.format](rows)


if __name__ == "__main__":
    main()
