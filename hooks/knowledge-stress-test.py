#!/usr/bin/env python3
"""
Stress test for knowledge pipeline.

Pulls real conversations from HuggingFace, converts to Claude Code
transcript format, and runs the extraction pipeline to measure:
  - Extraction rate (insights per conversation)
  - Dedup accuracy (unique vs corroborated)
  - Performance (time per transcript)
  - Quality distribution (tier breakdown)
  - False positives (noise that slips through)

Uses a SEPARATE test database — never touches production knowledge.db.

Usage:
  python3 knowledge-stress-test.py                    # run full test suite
  python3 knowledge-stress-test.py --conversations 50 # custom count
  python3 knowledge-stress-test.py --dataset lmsys     # specific dataset
  python3 knowledge-stress-test.py --only-extract      # skip HF, use cached
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Use the knowledge-store plugin library
sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB, compute_insight_hash  # noqa: E402
from knowledge_lib.init_db import init_db  # noqa: E402

# Import the write hook's extraction functions
HOOKS_DIR = Path.home() / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

# We'll import extraction functions inline to avoid circular issues


# ── HuggingFace Dataset Loaders ──

DATASETS = {
    "lmsys": {
        "name": "lmsys/chatbot_arena_conversations",
        "split": "train",
        "desc": "Real human-LLM conversations from Chatbot Arena",
    },
    "oasst": {
        "name": "OpenAssistant/oasst1",
        "split": "train",
        "desc": "Human-assistant conversations from OpenAssistant",
    },
    "sharegpt": {
        "name": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "split": "train",
        "desc": "ShareGPT conversation dumps",
    },
}

CACHE_DIR = Path(tempfile.gettempdir()) / "knowledge-stress-test"


def load_conversations(dataset_key: str, max_convos: int = 100) -> List[List[dict]]:
    """Load conversations from HuggingFace and convert to transcript format."""
    from datasets import load_dataset

    ds_config = DATASETS.get(dataset_key)
    if not ds_config:
        print(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")
        sys.exit(1)

    print(f"Loading {ds_config['desc']}...")
    print(f"  Dataset: {ds_config['name']}")

    try:
        ds = load_dataset(ds_config["name"], split=ds_config["split"], streaming=True)
    except Exception as e:
        print(f"  Failed to load: {e}")
        print("  Falling back to synthetic data...")
        return generate_synthetic_conversations(max_convos)

    conversations = []
    for i, example in enumerate(ds):
        if i >= max_convos:
            break

        transcript = convert_to_transcript(example, dataset_key)
        if transcript and len(transcript) >= 5:
            conversations.append(transcript)

        if (i + 1) % 25 == 0:
            print(f"  Loaded {i + 1}/{max_convos} conversations...")

    print(f"  Got {len(conversations)} valid conversations")
    return conversations


def convert_to_transcript(example: dict, dataset_key: str) -> List[dict]:
    """Convert a HuggingFace conversation to Claude Code JSONL transcript format."""
    messages = []

    if dataset_key == "lmsys":
        # lmsys format: {"conversation_a": [{"role": "user", "content": "..."}, ...]}
        convo = example.get("conversation_a", example.get("conversation", []))
        if isinstance(convo, list):
            for msg in convo:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(make_transcript_entry(role, content))

    elif dataset_key == "oasst":
        # OASST format: flat messages with parent_id threading
        text = example.get("text", "")
        role = "assistant" if example.get("role") == "assistant" else "user"
        messages.append(make_transcript_entry(role, text))

    elif dataset_key == "sharegpt":
        # ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, ...]}
        convos = example.get("conversations", [])
        if isinstance(convos, list):
            for msg in convos:
                role = "assistant" if msg.get("from") in ("gpt", "assistant") else "user"
                content = msg.get("value", "")
                messages.append(make_transcript_entry(role, content))

    return messages


def make_transcript_entry(role: str, content: str) -> dict:
    """Create a Claude Code transcript JSONL entry."""
    return {
        "type": role,
        "message": {
            "role": role,
            "content": [{"type": "text", "text": content}],
        },
    }


def generate_synthetic_conversations(count: int) -> List[List[dict]]:
    """Generate synthetic conversations with known insight patterns for testing."""
    templates = [
        # Tier 1: explicit insight markers
        "★ Insight ─────────────────────────────────────\n"
        "The {tech} pattern requires careful handling of {concept} because {reason}.\n"
        "─────────────────────────────────────────────────",

        # Tier 2: discovery language
        "I discovered that {tech} actually {discovery} when you {context}.",
        "The root cause was {cause}. The fix is to {fix} instead of {wrong_approach}.",
        "This works because {tech} uses {mechanism} internally, which means {implication}.",

        # Tier 3: observations
        "Importantly, {tech} has a {property} that affects {outcome}.",
        "Note that {tech} differs from {alt} in how it handles {aspect}.",

        # Noise (should be filtered)
        "```python\ndef foo():\n    return bar\n```",
        "- item 1\n- item 2",
        "https://example.com/very/long/url/that/should/be/filtered",
    ]

    techs = ["React", "SQLite", "MCP protocol", "Claude API", "Next.js",
             "TypeScript", "Supabase", "Docker", "FTS5", "Pinecone"]
    concepts = ["state management", "connection pooling", "token limits",
                "error boundaries", "schema migrations", "rate limiting"]
    reasons = ["the event loop blocks on I/O", "the cache invalidates on write",
               "the index rebuilds are expensive", "the connection drops after 30s"]

    import random
    random.seed(42)  # Reproducible

    conversations = []
    for _ in range(count):
        messages = []
        # 5-15 messages per conversation
        n_msgs = random.randint(5, 15)
        for j in range(n_msgs):
            if j % 2 == 0:
                content = f"How does {random.choice(techs)} handle {random.choice(concepts)}?"
                messages.append(make_transcript_entry("user", content))
            else:
                template = random.choice(templates)
                content = template.format(
                    tech=random.choice(techs),
                    concept=random.choice(concepts),
                    reason=random.choice(reasons),
                    discovery=random.choice(reasons),
                    context=f"working with {random.choice(concepts)}",
                    cause=random.choice(reasons),
                    fix=f"use {random.choice(techs)}'s built-in {random.choice(concepts)}",
                    wrong_approach=f"manually managing {random.choice(concepts)}",
                    mechanism=random.choice(concepts),
                    implication=random.choice(reasons),
                    property=random.choice(concepts),
                    outcome=random.choice(reasons),
                    alt=random.choice(techs),
                    aspect=random.choice(concepts),
                )
                messages.append(make_transcript_entry("assistant", content))
        conversations.append(messages)

    print(f"  Generated {len(conversations)} synthetic conversations")
    return conversations


# ── Stress Test Runner ──

def write_transcript_file(messages: List[dict], path: Path) -> None:
    """Write messages as JSONL transcript file."""
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


def run_extraction_test(conversations: List[List[dict]], test_db_path: Path) -> dict:
    """Run the full extraction pipeline against a test database."""
    # Import extraction functions from the write hook
    # We need to do this carefully to avoid module conflicts
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "knowledge_write", HOOKS_DIR / "knowledge-write.py"
    )
    kw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kw)

    # Initialize test database
    conn = init_db(test_db_path)
    conn.close()
    db = KnowledgeDB(db_path=test_db_path)

    results = {
        "total_conversations": len(conversations),
        "total_messages": 0,
        "total_extracted": 0,
        "total_saved": 0,
        "total_corroborated": 0,
        "total_noise_filtered": 0,
        "tier_distribution": {1: 0, 2: 0, 3: 0},
        "extraction_times": [],
        "insights_per_convo": [],
        "errors": 0,
    }

    transcript_dir = CACHE_DIR / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    for i, convo in enumerate(conversations):
        # Write transcript to temp file
        transcript_path = transcript_dir / f"convo_{i:04d}.jsonl"
        write_transcript_file(convo, transcript_path)

        results["total_messages"] += len(convo)

        # Time the extraction
        t0 = time.time()
        try:
            insights = kw.extract_insights(convo)
        except Exception as e:
            results["errors"] += 1
            continue
        elapsed = time.time() - t0
        results["extraction_times"].append(elapsed)

        n_extracted = len(insights)
        results["total_extracted"] += n_extracted
        results["insights_per_convo"].append(n_extracted)

        # Store in test DB
        saved = 0
        corroborated = 0
        for ins in insights:
            text = ins["text"]
            h = compute_insight_hash(text)
            tier = ins.get("tier", 3)
            results["tier_distribution"][tier] = results["tier_distribution"].get(tier, 0) + 1

            existing = db.get_insight(h)
            if existing is not None:
                db.increment_corroboration(h)
                corroborated += 1
            else:
                row_id = db.upsert_insight(
                    text=text, tier=tier,
                    source="stress_test",
                    confidence=ins.get("confidence", 0.5),
                    session_id=f"stress_{i:04d}",
                    is_personal=ins.get("is_personal", False),
                )
                if row_id is not None:
                    saved += 1

        results["total_saved"] += saved
        results["total_corroborated"] += corroborated

        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(conversations)} conversations "
                  f"({results['total_extracted']} insights so far)")

    db.close()
    return results


def run_quality_checks(test_db_path: Path) -> dict:
    """Run quality checks on extracted insights."""
    db = KnowledgeDB(db_path=test_db_path)

    rows = db.fetchall("SELECT * FROM insights WHERE expired_at IS NULL")
    checks = {
        "total_insights": len(rows),
        "avg_length": 0,
        "too_short": 0,   # < 40 chars
        "too_long": 0,    # > 500 chars
        "has_code": 0,    # contains ``` or def/function
        "has_url": 0,     # contains http
        "has_sensitive": 0,  # potential secrets
        "duplicates": 0,
    }

    texts = []
    for row in rows:
        r = dict(row)
        text = r.get("text", "")
        texts.append(text)
        checks["avg_length"] += len(text)

        if len(text) < 40:
            checks["too_short"] += 1
        if len(text) > 500:
            checks["too_long"] += 1
        if "```" in text or "def " in text or "function " in text:
            checks["has_code"] += 1
        if "http" in text[:50]:
            checks["has_url"] += 1
        if any(p in text.lower() for p in ["api_key", "sk-", "password"]):
            checks["has_sensitive"] += 1

    if texts:
        checks["avg_length"] = checks["avg_length"] // len(texts)

    # Check for duplicates (same first 60 chars)
    prefixes = [t[:60].lower() for t in texts]
    checks["duplicates"] = len(prefixes) - len(set(prefixes))

    db.close()
    return checks


def print_report(results: dict, quality: dict, elapsed: float):
    """Print the stress test report."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE PIPELINE STRESS TEST REPORT")
    print("=" * 60)

    print(f"\nDataset:")
    print(f"  Conversations:    {results['total_conversations']}")
    print(f"  Total messages:   {results['total_messages']}")
    print(f"  Wall time:        {elapsed:.1f}s")

    print(f"\nExtraction:")
    print(f"  Insights found:   {results['total_extracted']}")
    print(f"  Saved (new):      {results['total_saved']}")
    print(f"  Corroborated:     {results['total_corroborated']}")
    print(f"  Errors:           {results['errors']}")

    if results['total_conversations'] > 0:
        rate = results['total_extracted'] / results['total_conversations']
        print(f"  Rate:             {rate:.1f} insights/conversation")

    if results['extraction_times']:
        avg_time = sum(results['extraction_times']) / len(results['extraction_times'])
        max_time = max(results['extraction_times'])
        p95_idx = int(len(results['extraction_times']) * 0.95)
        sorted_times = sorted(results['extraction_times'])
        p95_time = sorted_times[min(p95_idx, len(sorted_times) - 1)]
        print(f"  Avg extract time: {avg_time*1000:.1f}ms")
        print(f"  P95 extract time: {p95_time*1000:.1f}ms")
        print(f"  Max extract time: {max_time*1000:.1f}ms")

    print(f"\nTier distribution:")
    for tier, count in sorted(results['tier_distribution'].items()):
        label = {1: "explicit", 2: "discovery", 3: "observation"}.get(tier, f"tier-{tier}")
        print(f"  {label}: {count}")

    print(f"\nQuality checks:")
    print(f"  Total in DB:      {quality['total_insights']}")
    print(f"  Avg length:       {quality['avg_length']} chars")
    print(f"  Too short (<40):  {quality['too_short']}")
    print(f"  Too long (>500):  {quality['too_long']}")
    print(f"  Contains code:    {quality['has_code']}")
    print(f"  Contains URL:     {quality['has_url']}")
    print(f"  Sensitive data:   {quality['has_sensitive']}")
    print(f"  Duplicates:       {quality['duplicates']}")

    # Verdicts
    print(f"\nVerdicts:")
    issues = []
    if quality['has_sensitive'] > 0:
        issues.append(f"FAIL: {quality['has_sensitive']} insights contain sensitive data")
    if quality['duplicates'] > quality['total_insights'] * 0.1:
        issues.append(f"WARN: {quality['duplicates']} duplicates ({100*quality['duplicates']/max(1,quality['total_insights']):.0f}%)")
    if quality['has_code'] > quality['total_insights'] * 0.05:
        issues.append(f"WARN: {quality['has_code']} insights contain code snippets")
    if quality['has_url'] > quality['total_insights'] * 0.05:
        issues.append(f"WARN: {quality['has_url']} insights contain URLs")
    if not issues:
        print("  ALL CHECKS PASSED")
    for issue in issues:
        print(f"  {issue}")


def main():
    parser = argparse.ArgumentParser(description="Stress test knowledge pipeline")
    parser.add_argument("--conversations", type=int, default=100,
                        help="Number of conversations to process (default 100)")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["synthetic"],
                        default="synthetic",
                        help="Dataset to use (default: synthetic)")
    parser.add_argument("--keep-db", action="store_true",
                        help="Keep test database after run")
    args = parser.parse_args()

    print("Knowledge Pipeline Stress Test")
    print(f"  Dataset: {args.dataset}")
    print(f"  Conversations: {args.conversations}")
    print()

    # Create isolated test database
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    test_db_path = CACHE_DIR / "stress_test.db"
    if test_db_path.exists():
        test_db_path.unlink()

    # Load data
    if args.dataset == "synthetic":
        conversations = generate_synthetic_conversations(args.conversations)
    else:
        conversations = load_conversations(args.dataset, args.conversations)

    if not conversations:
        print("No conversations loaded. Exiting.")
        sys.exit(1)

    # Run extraction
    print(f"\nRunning extraction pipeline...")
    t0 = time.time()
    results = run_extraction_test(conversations, test_db_path)
    elapsed = time.time() - t0

    # Run quality checks
    quality = run_quality_checks(test_db_path)

    # Report
    print_report(results, quality, elapsed)

    # Cleanup
    if not args.keep_db:
        if test_db_path.exists():
            test_db_path.unlink()
        print(f"\nTest database cleaned up.")
    else:
        print(f"\nTest database kept at: {test_db_path}")

    print(f"Transcript cache at: {CACHE_DIR / 'transcripts'}")


if __name__ == "__main__":
    main()
