#!/usr/bin/env python3
"""
Stop hook: extract insights from conversation transcript → knowledge.db

Karpathy principle: every conversation is a data collection opportunity.
Each insight is a labeled training example with provenance:
  - tier (quality label: 0=pinned, 1=explicit marker, 2=discovery, 3=observation)
  - confidence (extraction confidence score)
  - salience (how important was this topic in the conversation?)
  - corroboration_count (how many sessions independently rediscovered this?)
  - votes (human preference signal)
  - project (domain tag)
  - session_id (provenance: which conversation produced this)

The ranking formula (tier * votes * recency * salience) is a reward model.
Corroboration is self-consistency across sessions. Voting is RLHF.
This isn't just a knowledge store — it's a preference dataset.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Safety: early exits ──

if not Path("/Volumes/OWC drive").exists():
    sys.exit(0)

HOOKS_DIR = Path.home() / ".claude" / "hooks"
KNOWLEDGE_DIR = Path("/Volumes/OWC drive/Knowledge")

if (HOOKS_DIR / "knowledge-write.disabled").exists():
    sys.exit(0)

sys.path.insert(0, str(Path.home() / ".claude" / "plugins" / "knowledge-store"))

from knowledge_lib.db import KnowledgeDB, compute_insight_hash  # noqa: E402

# ── Logging ──

LOG_FILE = KNOWLEDGE_DIR / "hooks" / "hook_log.jsonl"


def log_entry(data: dict):
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
        # Periodic trim: keep last 100 lines when file exceeds 150
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 20000:
            lines = LOG_FILE.read_text().strip().splitlines()
            if len(lines) > 150:
                LOG_FILE.write_text("\n".join(lines[-100:]) + "\n")
    except IOError:
        pass


# ── Sensitive data filtering ──

# Patterns split with concatenation to avoid tripping secret scanners
_SK = "sk-"
_SK_ANT = "sk-" + "ant-"
_GH = "gh" + "[pousr]_"
_XOX = "xox" + "[baprs]-"
_PK = "PRI" + "VATE KEY"

SENSITIVE_PATTERNS = [
    re.compile(p) for p in [
        r'(?i)(?:api[_-]?key|apikey|api_secret)["\s:=]+["\']?([a-zA-Z0-9_\-]{16,})',
        r'(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}',
        r'(?<![a-zA-Z])' + _SK + r'[a-zA-Z0-9\-]{20,}',
        r'(?<![a-zA-Z])' + _SK_ANT + r'[a-zA-Z0-9\-]{10,}',
        _GH + r'[A-Za-z0-9_]{20,}',
        r'(?i)(?:bearer|token|secret|password)["\s:=]+["\']?([a-zA-Z0-9_\-\.]{8,})',
        r'-----BEGIN (?:RSA |EC |OPENSSH )?' + _PK + r'-----',
        r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
        r'(?i)(?:mongodb|postgres|mysql|redis)://[^\s"\']+:[^\s"\']+@',
        _XOX + r'[0-9]{10,}-[0-9]{10,}-[a-zA-Z0-9]{20,}',
    ]
]


def contains_sensitive(text: str) -> bool:
    return any(p.search(text) for p in SENSITIVE_PATTERNS)


def redact(text: str) -> str:
    result = text
    for p in SENSITIVE_PATTERNS:
        result = p.sub("[REDACTED]", result)
    return result


# ── Transcript reading ──

ALLOWED_DIRS = [
    Path.home() / ".claude" / "projects",
    Path("/tmp"),
    Path("/var/folders"),
]


def is_safe_path(path_str: str) -> bool:
    try:
        path = Path(path_str).resolve()
        return any(path.is_relative_to(d.resolve()) for d in ALLOWED_DIRS)
    except (OSError, ValueError):
        return False


def read_transcript(path: str, max_lines: int = 0) -> List[dict]:
    messages = []  # type: List[dict]
    try:
        with open(path, "r") as f:
            lines = deque(f, maxlen=max_lines) if max_lines > 0 else f
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        pass
    return messages


def get_text(msg: dict) -> str:
    """Extract text content from a transcript message."""
    if "message" in msg and isinstance(msg["message"], dict):
        blocks = msg["message"].get("content", [])
        if isinstance(blocks, list):
            return " ".join(
                b.get("text", "") for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
    if "content" in msg and isinstance(msg["content"], str):
        return msg["content"]
    return ""


# ── Quality filters ──

def is_good_sentence(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        return False
    if text[0] in "({[<`":
        return False
    verbs = [" is ", " are ", " was ", " has ", " have ", " can ", " will ",
             " should ", " means ", " works ", " requires ", " prevents "]
    return text[-1] in '.!?":' or any(v in text.lower() for v in verbs)


def is_noise(text: str) -> bool:
    text = text.strip()
    if any(c in text for c in "├└┌┐│─═║"):
        return True
    if text.startswith("`") or text.startswith("```"):
        return True
    if text[:2] in ("- ", "* ") and len(text) <= 52:
        return True
    code = ["```", "()", "{}", "=>", "->", "def ", "function ", "import "]
    if sum(1 for m in code if m in text) >= 3:
        return True
    if "http" in text[:50] or text.count("{") > 2:
        return True
    return False


# ── Insight extraction (3-tier regex) ──
#
# This is the "labeling function" in Karpathy terms.
# Tier 1 = explicit markers (high precision, low recall)
# Tier 2 = discovery language (medium precision)
# Tier 3 = observations (lower precision, higher recall)
#
# Each tier maps to a confidence score. Combined with salience
# and votes, this creates a preference-labeled dataset.

def _compile_tiers():
    """Pre-compile tier patterns at module load time."""
    _base_flags = re.IGNORECASE | re.MULTILINE
    raw = [
        (1, [
            (r"★\s*Insight[─\s\n]*(.*?)(?=─{5,}|$)", True),
            (r"(?:Key (?:insight|finding|takeaway)|Important):\s*(.+?)(?:\n\n|$)", False),
        ]),
        (2, [
            (r"(?:discovered|found|realized|learned)\s+that\s+(.{40,250}?)(?:\.|!|\n\n)", False),
            (r"(?:root cause|the (?:issue|problem|bug) (?:is|was)|solution is|fix (?:is|was))\s*[:\-]?\s*(.{30,300}?)(?:\.|!|\n\n)", False),
            (r"(?:this (?:means|indicates|works because)|the reason is)\s+(.{40,250}?)(?:\.|!|\n\n)", False),
            (r"[Tt]his works (?:by|because)\s+(.{40,250}?)(?:\.|!|\n\n)", False),
        ]),
        (3, [
            (r"(?:importantly|notably|significantly)[,:\s]+(.{40,250}?)(?:\.|!|\n\n)", False),
            (r"[Nn]ote that\s+(.{40,250}?)(?:\.|!|\n\n)", False),
            (r"(.{40,250}?(?:trade-off|tradeoff).{0,150}?)(?:\.|!|\n\n)", False),
        ]),
    ]
    compiled = []
    for tier, patterns in raw:
        cp = []
        for pattern, use_dotall in patterns:
            flags = _base_flags | re.DOTALL if use_dotall else _base_flags
            cp.append(re.compile(pattern, flags))
        compiled.append((tier, cp))
    return compiled


TIERS = _compile_tiers()

PERSONAL_INDICATORS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\byou(r|'re|'ve)?\b", r"\bproject(s)?\b", r"\bcodebase\b",
        r"\bworkflow\b", r"\bpatterns?\b.*\byour\b", r"\bapproach\b",
    ]
]


def extract_insights(messages: List[dict]) -> List[dict]:
    insights = []  # type: List[dict]
    seen = set()  # type: set

    for msg in messages:
        message = msg.get("message", {})
        blocks = message.get("content", [])
        if not blocks and "content" in msg:
            c = msg.get("content", "")
            if isinstance(c, str):
                blocks = [{"type": "text", "text": c}]
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            content = block.get("text", "")
            if not isinstance(content, str) or len(content) < 50:
                continue
            content = content[:15000]

            for tier, patterns in TIERS:
                for compiled_pat in patterns:
                    for match in compiled_pat.findall(content):
                        text = match.strip() if isinstance(match, str) else str(match).strip()
                        if len(text) < 40 or len(text) > 500:
                            continue
                        if contains_sensitive(text):
                            continue
                        if tier == 1:
                            if text[0] in "`>*" or not any(c.isalpha() for c in text[:5]):
                                continue
                        elif is_noise(text) or not is_good_sentence(text):
                            continue

                        key = text[:60].lower()
                        if key in seen:
                            continue
                        seen.add(key)

                        text = re.sub(r"\s+", " ", text).strip()
                        text = redact(text)
                        personal = any(p.search(text) for p in PERSONAL_INDICATORS)
                        insights.append({
                            "text": text,
                            "tier": tier - 1 if personal else tier,
                            "is_personal": personal,
                            "confidence": 0.5 + (3 - tier) * 0.15 + (0.1 if personal else 0),
                        })

    insights.sort(key=lambda x: (x["tier"], not x.get("is_personal", False)))
    return insights


# ── Salience: how important was this topic in the conversation? ──

_SALIENCE_STOP_WORDS = {
    "that", "this", "with", "from", "have", "been", "were", "they",
    "their", "which", "about", "would", "there", "these", "other",
    "into", "more", "when", "some", "than", "them", "also", "just",
}

_SALIENCE_MARKERS = [
    re.compile(m, re.IGNORECASE) for m in [
        r"\bkey insight\b", r"\bimportant\b", r"\bcritical\b",
        r"\broot cause\b", r"\bfundamental\b", r"\bbreakthrough\b",
    ]
]


def precompute_salience_context(messages: List[dict]) -> tuple:
    """Pre-compute shared context for salience scoring. Call once per session."""
    all_text = [get_text(m) for m in messages[-50:]]
    all_text_lower = [t.lower() for t in all_text if t]
    full_context = " ".join(all_text_lower)
    marker_count = sum(1 for m in _SALIENCE_MARKERS if m.search(full_context))
    return all_text_lower, marker_count


def compute_salience(text: str, ctx: tuple, tier: int) -> float:
    all_text_lower, marker_count = ctx
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    keywords = [w for w in words if w not in _SALIENCE_STOP_WORDS][:5]
    if not keywords:
        return 0.3

    score = 0.0

    # Topic persistence: what fraction of messages mention these keywords?
    hits = sum(1 for t in all_text_lower if any(kw in t for kw in keywords))
    score += min(0.4, (hits / max(1, len(all_text_lower))) * 2.0)

    # Importance markers (pre-computed)
    score += min(0.2, marker_count * 0.05)

    # Tier bonus
    score += {1: 0.2, 2: 0.1, 3: 0.0}.get(tier, 0.0)

    return min(1.0, max(0.1, score))


# ── Project extraction ──

def extract_project(cwd: str) -> Optional[str]:
    if not cwd:
        return None
    p = Path(cwd)
    for dev_root in ["/Volumes/OWC drive/Dev", str(Path.home())]:
        try:
            rel = p.relative_to(dev_root)
            return rel.parts[0] if rel.parts else None
        except ValueError:
            continue
    return p.name or None


# ── Main ──

def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path", "")
    event_name = hook_input.get("hook_event_name", "unknown")

    # Skip subagent events (fires per-subagent, causing duplicates)
    if event_name in ("SubagentStop",):
        sys.exit(0)

    # Rate limit: same session within 5 minutes (compaction re-fires Stop)
    rate_file = HOOKS_DIR / ".knowledge-write-last.json"
    try:
        if rate_file.exists():
            rd = json.loads(rate_file.read_text())
            if rd.get("sid") == session_id and time.time() - rd.get("ts", 0) < 300:
                sys.exit(0)
    except (json.JSONDecodeError, IOError):
        pass

    # Validate transcript
    if not transcript_path or not Path(transcript_path).exists():
        sys.exit(0)
    if not is_safe_path(transcript_path):
        log_entry({"level": "error", "msg": f"rejected path: {transcript_path}",
                    "ts": datetime.utcnow().isoformat()})
        sys.exit(0)

    # Read transcript (tail for large files)
    try:
        file_size = Path(transcript_path).stat().st_size
    except OSError:
        sys.exit(0)
    max_lines = 300 if file_size > 10_000_000 else 0
    messages = read_transcript(transcript_path, max_lines=max_lines)
    if len(messages) < 5:
        sys.exit(0)

    # Extract & store
    insights = extract_insights(messages)
    project = extract_project(hook_input.get("cwd", ""))

    db = KnowledgeDB()
    saved = 0
    corroborated = 0

    # Pre-compute: batch hash existence check + salience context
    salience_ctx = precompute_salience_context(messages)
    hashes = [compute_insight_hash(ins["text"]) for ins in insights]
    if hashes:
        placeholders = ",".join("?" for _ in hashes)
        existing_rows = db.fetchall(
            f"SELECT hash FROM insights WHERE hash IN ({placeholders})", tuple(hashes)
        )
        existing_hashes = {r["hash"] for r in existing_rows}
    else:
        existing_hashes = set()

    for ins, h in zip(insights, hashes):
        text = ins["text"]
        if h in existing_hashes:
            db.increment_corroboration(h)
            corroborated += 1
            continue

        salience = compute_salience(text, salience_ctx, ins["tier"])
        row_id = db.upsert_insight(
            text=text,
            tier=ins["tier"],
            source="session_hook",
            confidence=ins.get("confidence", 0.5),
            session_id=session_id,
            project=project,
            is_personal=ins.get("is_personal", False),
            salience=salience,
        )
        if row_id is not None:
            saved += 1

    # Periodic pruning: expire stale low-tier insights
    pruned = db.prune_stale(max_age_days=90, min_tier=3)
    db.close()

    # Log
    log_entry({
        "session_id": session_id, "event": event_name,
        "messages": len(messages), "extracted": len(insights),
        "saved": saved, "corroborated": corroborated, "pruned": pruned,
        "project": project, "ts": datetime.utcnow().isoformat(),
    })

    # Rate-limit tracker
    try:
        rate_file.write_text(json.dumps({"sid": session_id, "ts": time.time()}))
    except IOError:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
