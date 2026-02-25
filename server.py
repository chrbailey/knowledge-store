"""knowledge-store MCP server — 5 tools for searchable knowledge base."""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, Optional

from knowledge_lib.db import KnowledgeDB

# =============================================================================
# SINGLETON STATE
# =============================================================================

_db = None  # type: Optional[KnowledgeDB]


def get_db() -> KnowledgeDB:
    global _db
    if _db is None:
        _db = KnowledgeDB()
    return _db


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

TOOLS = [
    {
        "name": "search_knowledge",
        "description": "Search captured session insights using FTS5 full-text search with vote-weighted ranking. Returns insights sorted by relevance, tier quality, and community votes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (keywords, natural language)"},
                "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                "project": {"type": "string", "description": "Filter by project name (e.g. 'promptspeak', 'touchgrass')"},
                "tier_max": {"type": "integer", "description": "Max tier to include (0=pinned, 1=discovery, 2=general, 3=observation). Default 3.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_insight",
        "description": "Get a single insight by its hash (16-char hex) or numeric ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Insight hash or numeric ID"},
            },
            "required": ["ref"],
        },
    },
    {
        "name": "vote_insight",
        "description": "Upvote or downvote an insight. Votes affect search ranking — upvoted insights surface higher.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string", "description": "Insight hash or numeric ID"},
                "vote": {"type": "string", "enum": ["up", "down"], "description": "Vote direction"},
            },
            "required": ["ref", "vote"],
        },
    },
    {
        "name": "pin_knowledge",
        "description": "Pin text as a tier-0 high-confidence insight. Pinned insights rank highest in search and SessionStart context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The insight text to pin"},
                "project": {"type": "string", "description": "Optional project context"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "knowledge_stats",
        "description": "Aggregate knowledge base stats: total insights, breakdown by tier/source/project, top voted insights.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

# =============================================================================
# TOOL HANDLERS
# =============================================================================


def handle_search_knowledge(args: dict) -> dict:
    db = get_db()
    results = db.search(
        query=args["query"],
        limit=args.get("limit", 10),
        tier_max=args.get("tier_max", 3),
        project=args.get("project"),
    )
    return {
        "count": len(results),
        "results": [
            {
                "id": r["id"],
                "hash": r["hash"],
                "text": r["text"],
                "tier": r["tier"],
                "source": r["source"],
                "project": r["project"],
                "upvotes": r["upvotes"],
                "downvotes": r["downvotes"],
                "captured_at": r["captured_at"],
            }
            for r in results
        ],
    }


def handle_get_insight(args: dict) -> dict:
    db = get_db()
    result = db.get_insight(args["ref"])
    if not result:
        return {"error": f"Insight not found: {args['ref']}"}
    return result


def handle_vote_insight(args: dict) -> dict:
    db = get_db()
    try:
        updated = db.vote(args["ref"], args["vote"])
        return {
            "status": "ok",
            "hash": updated["hash"],
            "upvotes": updated["upvotes"],
            "downvotes": updated["downvotes"],
            "net_votes": updated["upvotes"] - updated["downvotes"],
        }
    except ValueError as e:
        return {"error": str(e)}


def handle_pin_knowledge(args: dict) -> dict:
    db = get_db()
    result = db.pin(args["text"], project=args.get("project"))
    return {
        "status": "pinned",
        "hash": result["hash"],
        "text": result["text"],
        "tier": result["tier"],
        "confidence": result["confidence"],
    }


def handle_knowledge_stats(args: dict) -> dict:
    db = get_db()
    return db.stats()


HANDLERS = {
    "search_knowledge": handle_search_knowledge,
    "get_insight": handle_get_insight,
    "vote_insight": handle_vote_insight,
    "pin_knowledge": handle_pin_knowledge,
    "knowledge_stats": handle_knowledge_stats,
}


# =============================================================================
# MCP JSON-RPC 2.0 PROTOCOL (copied from touchgrass server.py)
# =============================================================================


def handle_request(request: dict) -> Optional[dict]:
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        client_version = params.get("protocolVersion", "2024-11-05")
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": client_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "knowledge-store", "version": "0.1.0"},
            },
        }

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        handler = HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }
        try:
            result = handler(tool_args)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            }

    return {
        "jsonrpc": "2.0", "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _read_message() -> Optional[dict]:
    """Read one MCP message from stdin using binary I/O for correct Content-Length handling."""
    global _use_content_length

    header_line = sys.stdin.buffer.readline()
    if not header_line:
        return None

    header_str = header_line.decode("utf-8", errors="replace").strip()
    if not header_str:
        return None

    # Content-Length framing (MCP SDK transport / Claude Code)
    if header_str.startswith("Content-Length:"):
        _use_content_length = True
        content_length = int(header_str.split(":", 1)[1].strip())
        while True:
            next_line = sys.stdin.buffer.readline()
            if not next_line or next_line.strip() == b"":
                break
        body = sys.stdin.buffer.read(content_length)
        return json.loads(body.decode("utf-8"))

    # Line-delimited JSON fallback
    return json.loads(header_str)


_use_content_length = False


def _write_message(response: dict) -> None:
    """Write one MCP message to stdout, matching client's framing style."""
    body = json.dumps(response).encode("utf-8")
    if _use_content_length:
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        sys.stdout.buffer.write(header + body)
    else:
        sys.stdout.buffer.write(body + b"\n")
    sys.stdout.buffer.flush()


def main():
    """Run the MCP server on stdin/stdout."""
    while True:
        try:
            request = _read_message()
            if request is None:
                break

            response = handle_request(request)
            if response is not None:
                _write_message(response)

        except json.JSONDecodeError:
            continue
        except Exception as e:
            sys.stderr.write(f"knowledge-store server error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
