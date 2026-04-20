# Contributing

Thanks for looking.

## Before opening a PR

1. **Open an issue first** for anything larger than a typo.
2. **All changes need tests.** If tests don't exist yet, at minimum add a test alongside your change.
3. **Match the existing code style.** Python 3.9 compatibility is required (use `Union[str, Path]`, not `str | Path`).
4. **Run the full test suite locally** before submitting.

## What this project will not accept

Knowledge Store is local-first by design. It reads Claude Code session transcripts, extracts insights, and stores them in a local SQLite database. Nothing leaves the user's machine. PRs that break this posture will not land.

- PRs that add outbound network calls from any hook or MCP tool. The entire value proposition is that conversations stay local.
- PRs that remove or weaken the 3-tier regex extraction pipeline (explicit markers, discovery language, observations). Feel free to improve precision or recall, but the tier labeling is load-bearing — it's what makes the extracted dataset usable for preference labeling.
- PRs that alter the quality score formula without accompanying benchmark results. Changing the weights changes what surfaces in future sessions.
- PRs that bypass deduplication (hash-based) or corroboration counting. Duplicate suppression and corroboration are how we separate signal from noise across thousands of sessions.
- PRs that add training-data export without an explicit user-triggered command. The user should always decide when to export.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not file security issues in the public tracker.

## Author

[Christopher Bailey](https://github.com/chrbailey).
