#!/usr/bin/env python3
"""
Validate knowledge.db insights against outside sources.

Karpathy's data curation principle: a smaller, cleaner dataset beats
a larger, noisier one. This script processes insights in batches:

  1. Pick N unvalidated tier 2-3 insights (lowest quality first)
  2. For each: web search for corroboration
  3. Score: PROMOTE (→ tier 1), KEEP (stay), EXPIRE (remove)
  4. Log every decision for audit

The validation itself happens externally (Claude agents do the search).
This script handles the bookkeeping: what to validate, how to record results.

Usage:
  # Get next batch of insights to validate
  python3 knowledge-validate.py --batch 10

  # Record a validation result
  python3 knowledge-validate.py --record <hash> --verdict promote --evidence "URL or citation"
  python3 knowledge-validate.py --record <hash> --verdict keep
  python3 knowledge-validate.py --record <hash> --verdict expire --reason "outdated"

  # Status report
  python3 knowledge-validate.py --status
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB  # noqa: E402

VALIDATION_LOG = Path("/Volumes/OWC drive/Knowledge/validation_log.jsonl")


def _read_validation_log() -> list:
    """Parse the validation log JSONL file. Returns list of entry dicts."""
    entries = []
    if VALIDATION_LOG.exists():
        for line in VALIDATION_LOG.read_text().strip().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def get_batch(db: KnowledgeDB, limit: int = 10) -> list:
    """Get next batch of unvalidated insights, prioritized by:
    1. Tier 3 before tier 2 (lowest quality first)
    2. Highest salience within tier (most likely to be promotable)
    3. Not yet validated (no validation_log entry)
    """
    validated = {e.get("hash") for e in _read_validation_log()}

    rows = db.fetchall(
        """SELECT id, hash, text, tier, confidence, salience, project,
                  upvotes, downvotes, corroboration_count, captured_at
           FROM insights
           WHERE expired_at IS NULL AND tier >= 2
           ORDER BY tier DESC, salience DESC, captured_at DESC"""
    )

    batch = []
    for row in rows:
        r = dict(row)
        if r["hash"] in validated:
            continue
        batch.append(r)
        if len(batch) >= limit:
            break

    return batch


def record_verdict(db: KnowledgeDB, insight_hash: str, verdict: str,
                   evidence: str = "", reason: str = ""):
    """Record a validation verdict and update the insight."""
    insight = db.get_insight(insight_hash)
    if not insight:
        print(f"Error: insight {insight_hash} not found")
        sys.exit(1)

    # Apply verdict
    if verdict == "promote":
        db.promote(insight_hash)
    elif verdict == "expire":
        db.expire(insight_hash)
    # "keep" = no DB change

    # Log the decision
    log_entry = {
        "hash": insight_hash,
        "verdict": verdict,
        "evidence": evidence,
        "reason": reason,
        "original_tier": insight.get("tier"),
        "text_preview": insight.get("text", "")[:100],
        "ts": datetime.utcnow().isoformat(),
    }
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"{verdict.upper()}: {insight.get('text', '')[:80]}...")


def print_status(db: KnowledgeDB):
    """Validation progress report."""
    total_validatable = db.fetchall(
        "SELECT COUNT(*) as cnt FROM insights WHERE expired_at IS NULL AND tier >= 2"
    )[0]["cnt"]

    entries = _read_validation_log()
    validated = len(entries)
    verdicts = {"promote": 0, "keep": 0, "expire": 0}
    for entry in entries:
        v = entry.get("verdict", "")
        if v in verdicts:
            verdicts[v] += 1

    remaining = total_validatable - validated
    print(f"Validation Progress")
    print(f"  Total to validate:  {total_validatable}")
    print(f"  Validated:          {validated}")
    print(f"  Remaining:          {remaining}")
    print(f"  Promoted → tier 1:  {verdicts['promote']}")
    print(f"  Kept at tier 2-3:   {verdicts['keep']}")
    print(f"  Expired:            {verdicts['expire']}")
    if validated > 0:
        pct = 100 * validated / total_validatable
        print(f"  Progress:           {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Validate knowledge.db insights")
    parser.add_argument("--batch", type=int, help="Get next N insights to validate")
    parser.add_argument("--record", help="Record verdict for insight hash")
    parser.add_argument("--verdict", choices=["promote", "keep", "expire"])
    parser.add_argument("--evidence", default="", help="URL or citation for promotion")
    parser.add_argument("--reason", default="", help="Reason for expiry")
    parser.add_argument("--status", action="store_true", help="Show validation progress")
    parser.add_argument("--json", action="store_true", help="Output as JSON (for agent consumption)")
    args = parser.parse_args()

    db = KnowledgeDB()

    if args.status:
        print_status(db)
    elif args.batch:
        batch = get_batch(db, limit=args.batch)
        if args.json:
            print(json.dumps(batch, indent=2, default=str))
        else:
            for i, row in enumerate(batch, 1):
                print(f"\n[{i}] hash={row['hash']}  tier={row['tier']}  "
                      f"salience={row.get('salience', 'n/a')}  project={row.get('project', 'none')}")
                print(f"    {row['text'][:200]}")
            print(f"\nTotal: {len(batch)} insights ready for validation")
    elif args.record and args.verdict:
        record_verdict(db, args.record, args.verdict,
                       evidence=args.evidence, reason=args.reason)
    else:
        parser.print_help()

    db.close()


if __name__ == "__main__":
    main()
