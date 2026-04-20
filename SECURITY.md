# Security

## Responsible Disclosure

If you find a security issue, please do **not** file a public GitHub issue.

Email: chris.bailey@erp-access.com — include "SECURITY: knowledge-store" in the subject line.

Expect an acknowledgment within 72 hours.

## What this tool does

Knowledge Store is a Claude Code plugin that reads session transcripts (JSONL) from the user's local `~/.claude/projects/` directory, extracts insights using a 3-tier regex pipeline, and stores them in a local SQLite database (`knowledge.db`). A SessionStart hook injects the top-ranked insights as context into new sessions. An MCP server exposes 5 tools (`search_knowledge`, `get_insight`, `vote_insight`, `pin_knowledge`, `knowledge_stats`) for the host agent.

Everything runs locally. The only thing the plugin reads that is not part of itself is the user's own Claude Code transcripts.

## What this tool does NOT do

- It does not send transcripts, insights, or any conversation data to any remote service.
- It does not make any outbound network call from any hook or MCP tool.
- It does not share insights across users — the database is scoped to the local filesystem.
- It does not auto-export training data. Export happens only when the user explicitly runs `knowledge-export.py`.
- It does not modify the user's Claude Code transcripts. Reads only.

## Known Considerations

- `knowledge.db` accumulates whatever the extraction pipeline captures from transcripts, which may include secrets, names, or confidential project details that appeared in conversations. Treat the file as sensitive — it is a lossy but readable replay of your work.
- The 3-tier regex extraction is not a content filter. If you paste an API key into a Claude Code session and it shows up adjacent to an "Important:" marker, it may land in the database. Use the `knowledge-dedup.py` or manual SQL to purge if needed.
- The MCP server uses stdio. If you expose it over a network transport, any caller that reaches the port can read the entire local knowledge base.
- The exported training data formats (raw, chat, preference) contain full insight text including any sensitive content described above — scrub before sharing.

If you see evidence of any of the "does NOT do" items, that is a security issue — please report.
