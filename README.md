# knowledge-store

A local-first knowledge extraction and retrieval system for Claude Code. Every conversation becomes a data collection opportunity — insights are automatically extracted, scored, deduplicated, and surfaced in future sessions.

Built on the principle that a preference-labeled dataset emerges naturally from daily AI-assisted development: tier labels are quality scores, votes are RLHF signals, corroboration is self-consistency across sessions, and salience is the attention weight.

**Status:** v0.1 experimental, running in production on one machine. Core library (`knowledge_lib/`), MCP server (`server.py`) and hooks are test-covered. Hook paths and the default DB location are hardcoded for the author's workstation — see "Portability" and "What this is NOT" below before adopting.

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code Session                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  SessionStart ──► knowledge-read.py                     │
│                   │ Query top insights by project       │
│                   │ Inject as context (stdout → Claude) │
│                   ▼                                     │
│  ... conversation happens ...                           │
│                                                         │
│  Stop ──────────► knowledge-write.py                    │
│                   │ Read transcript (JSONL)             │
│                   │ 3-tier regex extraction             │
│                   │ Compute salience & confidence       │
│                   │ Deduplicate via hash                │
│                   │ Corroborate existing insights       │
│                   ▼                                     │
│               knowledge.db (SQLite + FTS5)              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Extraction Pipeline

Insights are extracted using a 3-tier regex system — a labeling function that trades off precision vs recall:

| Tier | Signal | Precision | Example Pattern |
|------|--------|-----------|-----------------|
| 1 | Explicit markers | High | `★ Insight`, `Key insight:`, `Important:` |
| 2 | Discovery language | Medium | `discovered that`, `root cause was`, `solution is` |
| 3 | Observations | Lower | `importantly`, `note that`, trade-off mentions |

Each extracted insight gets:
- **tier** — quality label (0=pinned, 1=explicit, 2=discovery, 3=observation)
- **confidence** — extraction confidence score (0.5 + tier bonus + personal bonus)
- **salience** — topic persistence across the conversation (keyword frequency + importance markers)
- **corroboration_count** — how many independent sessions rediscovered this
- **votes** — human preference signal (upvote/downvote via MCP tool)

### Quality Score Formula

```
score = 0.30 × tier_weight + 0.20 × vote_signal + 0.20 × corroboration + 0.15 × salience + 0.15 × confidence
```

### Search (FTS5 + Porter Tokenizer)

```
rank = fts_rank × tier_weight × (1 + 0.1 × net_votes + 0.05 × corroboration) × recency_decay × (0.5 + salience)
```

## What This Is NOT

- **Not cross-platform plug-and-play.** The default DB path is `/Volumes/OWC drive/Knowledge/knowledge.db` and several hook scripts early-exit when that path is missing. Expect to edit `knowledge_lib/init_db.py` and the hook scripts before this works on another machine. See "Portability" below.
- **Not a replacement for a hosted memory backend.** There is no sync, no multi-machine replication, no auth layer. One DB file per user.
- **Not semantic search.** Ranking is FTS5 BM25 plus preference/corroboration/recency/salience multipliers. Good at keyword + phrase recall; will miss paraphrases that share no tokens.
- **Not a general insight extractor.** The 3-tier regex set is tuned for Claude Code transcripts (JSONL from `~/.claude/projects/*`). Plugging a different transcript format in will require pattern tweaks.
- **Not an MCP "emotional memory" tool.** For mood / dual-engine / governance semantics see [MyPersona](https://github.com/chrbailey/MyPersona).

## Components

### Core Library (`knowledge_lib/`)

- **`db.py`** — `KnowledgeDB` class: upsert, search, vote, pin, prune, stats. SQLite + FTS5.
- **`init_db.py`** — Schema creation with migrations. 25+ column `insights` table, FTS5 virtual table, votes table.

### MCP Server (`server.py`)

5 tools exposed via raw JSON-RPC 2.0:

| Tool | Purpose |
|------|---------|
| `search_knowledge` | Full-text search with vote-weighted ranking |
| `get_insight` | Retrieve single insight by hash |
| `vote_insight` | Upvote/downvote (RLHF signal) |
| `pin_knowledge` | Pin text as tier-0 insight |
| `knowledge_stats` | Dataset health dashboard |

### Hooks (`hooks/`)

| Hook | Event | Purpose |
|------|-------|---------|
| `knowledge-read.py` | SessionStart | Inject top insights as context |
| `knowledge-write.py` | Stop | Extract insights from transcript |
| `knowledge-export.py` | Manual | Export as training data (raw, chat, preference) |
| `knowledge-validate.py` | Manual | Batch validation pipeline |
| `knowledge-dedup.py` | Manual | Deduplicate and expire stale insights |
| `knowledge-stress-test.py` | Manual | Stress test with synthetic + real data |

## Setup (60 seconds to tests green)

### 1. Clone and run the tests

```bash
git clone https://github.com/chrbailey/knowledge-store.git
cd knowledge-store
python3 -m venv .venv
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
```

Tests run against a temp DB — they do not touch any existing knowledge file.

### 2. Install as a Claude Code plugin (optional)

```bash
git clone https://github.com/chrbailey/knowledge-store.git ~/.claude/plugins/knowledge-store
```

### 3. Enable the plugin

In `~/.claude/settings.json`:

```json
{
  "enabledPlugins": {
    "knowledge-store@local": true
  }
}
```

### 4. Wire up hooks

Add to `~/.claude/settings.json` under `"hooks"`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "python3 -u ~/.claude/plugins/knowledge-store/hooks/knowledge-read.py",
        "timeout": 10000,
        "statusMessage": "Loading knowledge context..."
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "python3 -u ~/.claude/plugins/knowledge-store/hooks/knowledge-write.py",
        "timeout": 30000,
        "statusMessage": "Extracting insights..."
      }]
    }]
  }
}
```

### 5. Configure the database path

The database defaults to `/Volumes/OWC drive/Knowledge/knowledge.db` (the author's external drive). To change it, edit `DB_PATH` in `knowledge_lib/init_db.py`. The hook scripts also include an early-exit if that directory is missing — if you re-point the DB, update the guards in `hooks/knowledge-read.py` and `hooks/knowledge-write.py` accordingly.

## Portability

This repo is honest about being single-machine. Anything you'd have to change to run it on a new workstation:

1. `knowledge_lib/init_db.py` — `DB_PATH` constant.
2. `hooks/knowledge-read.py` — the `if not Path("/Volumes/OWC drive").exists()` guard.
3. `hooks/knowledge-write.py` — same guard plus transcript directory assumptions.
4. `hooks/knowledge-export.py`, `knowledge-dedup.py`, `knowledge-validate.py`, `knowledge-stress-test.py` — check each before running, same `/Volumes/OWC drive` assumption.

A future version may move these to an environment variable. Until it does: treat this as a reference implementation more than a drop-in plugin.

## Training Data Export

Export insights in three formats for fine-tuning or analysis:

```bash
# Raw JSONL — all fields
python3 hooks/knowledge-export.py --format raw --output insights-raw.jsonl

# Chat JSONL — OpenAI/Anthropic fine-tuning compatible
python3 hooks/knowledge-export.py --format chat --output insights-chat.jsonl

# Preference pairs — DPO/RLHF training
python3 hooks/knowledge-export.py --format preference --output insights-pref.jsonl

# Dataset health stats
python3 hooks/knowledge-export.py --stats
```

## Maintenance

```bash
# Deduplicate and clean
python3 hooks/knowledge-dedup.py --dry-run   # preview
python3 hooks/knowledge-dedup.py              # execute

# Validate tier 2-3 insights
python3 hooks/knowledge-validate.py --batch 10        # get batch
python3 hooks/knowledge-validate.py --status          # progress

# Stress test (isolated DB, never touches production)
python3 hooks/knowledge-stress-test.py
```

## Stress Test Results

Tested against real-world data:

| Dataset | Messages | Duration | Errors | Notes |
|---------|----------|----------|--------|-------|
| Real Claude Code transcripts | 45,000+ | 1.5s | 0 | 10 session files |
| ChatGPT conversations.json (350MB) | 200 conversations | 21s | 0 | 48M chars, cross-format |
| Synthetic edge cases | 1,000+ | <1s | 0 | Unicode, code blocks, secrets |

Security validation: 0 API keys leaked, 0 secrets in output, regex patterns catch `sk-*`, `AKIA*`, JWTs, connection strings, bearer tokens. Fixed false positive on words like "risk-free" via negative lookbehind.

## Known Limitations

- **66% fragment rate in practice.** The author's own production DB (~3,400 insights as of Apr 2026) is ~66% low-value sentence fragments when audited by hand. The extraction pipeline is precision-biased at tier 1, recall-biased at tiers 2-3. A future quality gate is planned.
- **Wiki-compiler is the primary polluter.** When chained with the author's wiki-compiler project, the extractor captures ~42% of its volume from that one source, with 86% of those being fragments. This is a per-project calibration issue, not a bug in the extractor, but worth knowing before enabling in a similar setup.
- **No privacy redaction beyond secrets regex.** The secrets patterns catch tokens and keys, but names, emails, and internal URLs are captured verbatim. Treat the DB file as you would any other private notebook.

## Comparison: Knowledge Store vs Open Brain

[Nate B Jones' Open Brain](https://github.com/nateb-jones) is a cloud-hosted second brain for AI assistants using Supabase + pgvector. Both solve the same problem — giving AI persistent memory across sessions — but make fundamentally different architectural bets.

### Architecture

| Dimension | Open Brain | Knowledge Store |
|-----------|-----------|-----------------|
| **Storage** | Supabase (hosted PostgreSQL + pgvector) | Local SQLite + FTS5 |
| **Search** | Vector similarity (semantic, 1536-dim embeddings) | Full-text search (BM25 + Porter tokenizer) |
| **Capture** | Slack webhook + MCP `save_memory` tool (manual) | SessionStart/Stop hooks (fully automatic) |
| **Retrieval** | MCP `search_memory` tool (on-demand) | SessionStart hook (automatic context injection) |
| **Cost** | ~$0.10–$0.30/month (Supabase free tier) + embedding API | $0 (fully local, no API calls) |
| **Privacy** | Cloud (Supabase hosted, data leaves machine) | Local disk only (never leaves machine) |
| **Multi-AI** | Any MCP client via remote URL | Claude Code only (hooks are Claude Code specific) |
| **Setup** | ~45 min, copy-paste friendly, non-technical | Python scripts, developer-oriented |
| **Quality signals** | None (flat storage) | 4-tier labeling, votes, corroboration, salience |
| **Deduplication** | None built-in | Hash-based dedup + prefix clustering |
| **Training export** | Not available | Raw, Chat (fine-tuning), Preference (DPO) formats |

### When to Use Which

**Choose Open Brain if:**
- You use multiple AI tools (Cursor, Windsurf, ChatGPT) and need shared memory
- You want semantic search ("things related to authentication" finds "OAuth flow" and "JWT tokens")
- You prefer hosted infrastructure with zero maintenance
- You're non-technical and want a guided setup
- Privacy of insights is not a primary concern

**Choose Knowledge Store if:**
- You use Claude Code as your primary AI tool
- You want zero-cost, zero-dependency, fully local operation
- You care about data quality (tiered extraction, deduplication, validation)
- You want training data export for fine-tuning or analysis
- Privacy is non-negotiable (nothing leaves your machine)
- You want automatic capture without manual "save this" commands

### The Fundamental Tradeoff

**Open Brain bets on breadth**: semantic search across any MCP client, cloud-hosted for accessibility, manual capture for precision.

**Knowledge Store bets on depth**: automatic extraction with quality scoring, preference signals that accumulate over time, training data that could feed back into model improvement. Every conversation is a data collection opportunity — not just storage, but a labeled dataset with provenance.

The approaches are complementary. Open Brain is a memory store. Knowledge Store is a preference dataset that happens to also be a memory store.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: [SECURITY.md](SECURITY.md). Changes: [CHANGELOG.md](CHANGELOG.md).

## License

MIT
