# Phase 7: MCP server

SupportPilot's support tools are available to any **Model Context Protocol** client (Claude Desktop, Claude Code,
Cursor, VS Code), so a support engineer can search the docs, file tickets and check ticket status from their own
AI assistant without opening the web app.

**Stack:** official `mcp` Python SDK 2.x (`MCPServer`, formerly FastMCP) · stdio and streamable-HTTP transports.
Code: [`src/supportpilot/mcp_server/server.py`](../../src/supportpilot/mcp_server/server.py).

## What it exposes
| Kind | Name | Annotations | What it does |
|---|---|---|---|
| Tool | `search_docs(query, top_k=5)` | read-only | Hybrid search (dense + BM25, RRF) over the SearchUnify docs; numbered passages with URLs |
| Tool | `create_ticket(customer_name, customer_email, subject, description)` | write, not destructive | Same `TicketService` as chat and the contact form: PII masking, DistilBERT classification, review flag, duplicate merge; `source=mcp` |
| Tool | `get_ticket_status(ticket_number, customer_email)` | read-only | Status, routing and latest support reply; scoped to the customer's email |
| Prompt | `answer_support_question(question)` | | Template: search, answer with `[n]` citations, offer a ticket if the docs don't cover it |

Every tool returns **structured output** (a JSON result plus an output schema, from Pydantic models), so clients
can use fields like `number` or `passages[].url` directly instead of parsing text.

## Design decisions
- **One write path.** MCP tickets go through `TicketService`, so they are masked, classified, de-duplicated,
  audited (`created` by `system`, then `classified` by `ai`) and appear in the staff dashboard like any other ticket.
- **Access model.** The server runs on the operator's machine (stdio, or HTTP bound to `127.0.0.1`), so it may
  file tickets for any customer. Reading a ticket still needs the customer's email, and "wrong email" and "no
  such ticket" give the same error, so a guessed number reveals nothing.
- **Annotations** (`read_only_hint`, `destructive_hint`) let clients auto-approve search and ask before creating
  a ticket.
- **Errors are tool errors** (`isError: true` with a readable message, e.g. which fields failed validation), so
  the calling model can correct itself instead of the session failing.
- **Robust launch.** MCP clients start servers from their own working directory; `__main__` moves to the project
  root first (or `SUPPORTPILOT_HOME`), so `.env`, the SQLite database and the classifier are found. Logs go to
  stderr; stdout carries only the protocol.

## Connect a client
Claude Desktop (`%APPDATA%\Claude\claude_desktop_config.json` on Windows), Cursor (`.cursor/mcp.json`) and
Claude Code (`.mcp.json`) all use the same shape:

```json
{
  "mcpServers": {
    "supportpilot": {
      "command": "C:\\path\\to\\AI_engineer_project\\.venv\\Scripts\\python.exe",
      "args": ["-m", "supportpilot.mcp_server"]
    }
  }
}
```

Claude Code one-liner: `claude mcp add supportpilot -- C:\path\to\.venv\Scripts\python.exe -m supportpilot.mcp_server`.
For HTTP clients: `python -m supportpilot.mcp_server --http` then connect to `http://127.0.0.1:8765/mcp`.
Qdrant must be running for `search_docs`.

## Verification
- **7 tests** (`tests/test_mcp_server.py`) through the SDK's in-process client, so the real protocol layer runs
  (schema generation, argument validation, error results): tool listing with annotations and output schemas,
  numbered passages, out-of-range `top_k` rejected before search runs, ticket created via `TicketService`
  (source, lower-cased email, card number masked, audit events), the same report sent twice merged into one
  ticket, validation errors listing every bad field, status scoped to the right email with no leak to another
  email, and the prompt. Full suite: **47 passed**.
- **stdio, real subprocess** launched from another directory, as Claude Desktop does: 3 tools listed; "How do I
  add Jira as a content source?" returned the Jira content-source pages and "Add a Content Source" from the
  live Qdrant index; an unknown ticket returned a clean tool error. Cold start to first result ≈ 11 s (imports
  and the embedding model load), later calls are fast.
- **Streamable HTTP**: listed tools and searched ("searches with no results" → the *Searches with No Result*
  analytics page first).
