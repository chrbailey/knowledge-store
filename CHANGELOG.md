# Changelog

All notable changes to knowledge-store are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- README now includes "What this is NOT", "Portability", and "Known Limitations"
  sections. The single-machine nature of the default paths is called out up front
  rather than buried in setup.
- Added this CHANGELOG.

### Fixed
- CI: `test_top_insights` now uses `days=3650` so the static fixture dates do not
  age out and cause intermittent failures.

## [0.1.0] — 2026-03-10

### Added
- SessionStart / Stop hooks that read from and write to the knowledge DB.
- Comparison table with Open Brain so readers can pick the right tool.
- Full README describing the 3-tier extraction, scoring formulas, and FTS5 ranking.

### Refactored
- Simplified DB interface, fixed correctness bugs, improved efficiency
  (commit `072cbe6`).

## [0.0.1] — 2026-02-25 (initial plugin)

### Added
- `KnowledgeDB` class over SQLite + FTS5 with Porter tokenizer.
- 25+ column `insights` table, `insights_fts` virtual table, `votes` table.
- MCP server exposing 5 tools (`search_knowledge`, `get_insight`, `vote_insight`,
  `pin_knowledge`, `knowledge_stats`).
- Hash-based deduplication.
- Upvote / downvote signal and corroboration tracking.
- Initial extraction pipeline targeting Claude Code transcripts.
