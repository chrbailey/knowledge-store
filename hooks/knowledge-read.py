#!/usr/bin/env python3
"""
SessionStart hook: inject relevant knowledge into context.

Thin wrapper around KnowledgeDB. All ranking logic lives in the library.
Output goes to stdout → Claude sees it as context.

Karpathy principle: the model's context window IS the training set for
this conversation. Curate what goes in like you'd curate training data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Safety: exit silently if OWC drive not mounted (DB lives there)
if not Path("/Volumes/OWC drive").exists():
    sys.exit(0)

sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB  # noqa: E402


def extract_project(cwd: str) -> str:
    """Derive project name from working directory."""
    if not cwd:
        return ""
    p = Path(cwd)
    for dev_root in ["/Volumes/OWC drive/Dev", str(Path.home())]:
        try:
            rel = p.relative_to(dev_root)
            return rel.parts[0] if rel.parts else ""
        except ValueError:
            continue
    return p.name


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    cwd = hook_input.get("cwd", "")
    project = extract_project(cwd)

    db = KnowledgeDB()
    bullets = []
    seen = set()

    # Project-scoped insights first (exact column match, not FTS keyword search)
    if project:
        for row in db.get_project_insights(project, limit=5, tier_max=2, days=30):
            h = row["hash"]
            if h not in seen:
                seen.add(h)
                bullets.append(row["text"][:200])

    # Global top insights (any project, recent high-quality)
    for row in db.get_top_insights(limit=5, days=14):
        h = row["hash"]
        if h not in seen:
            seen.add(h)
            bullets.append(row["text"][:200])

    # Knowledge gaps worth investigating
    investigation = db.get_investigation_prompts(project=project, limit=2)
    db.close()

    # Cap total context (~2000 chars, ~500 tokens)
    bullets = bullets[:10]

    if not bullets and not investigation:
        sys.exit(0)

    # Plain text to stdout → becomes Claude's context
    parts = []
    if bullets:
        parts.append("Prior insights:\n" + "\n".join(f"- {b}" for b in bullets))
    if investigation:
        parts.append("\n".join(investigation))

    print("\n\n".join(parts))
    sys.exit(0)


if __name__ == "__main__":
    main()
