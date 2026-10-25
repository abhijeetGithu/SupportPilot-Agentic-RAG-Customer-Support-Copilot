# Phase 5: FastAPI backend

## Endpoints (17 under `/api`, plus `/health`)
| Area | Endpoints |
|---|---|
| Chat | `POST /sessions` · `POST /chat/{sid}/stream` (SSE) · `POST /chat/{sid}/resume` (SSE) · `GET /chat/{sid}/history` · `GET /chat/{sid}/messages?after=` |
| Customer tickets | `POST /tickets` (contact form) · `POST /tickets/lookup` · `POST /tickets/{number}/replies` |
| Staff (JWT) | `GET /admin/tickets` (filters, search, priority order, pages) · `GET/PATCH /admin/tickets/{id}` · `POST /admin/tickets/{id}/replies` · `POST /admin/tickets/{id}/suggest-reply` · `GET /admin/escalations` · `POST /admin/escalations/{id}/take` · `GET /admin/metrics` · `POST /admin/classify` |
| Auth | `POST /auth/login` |

## Streaming (SSE)
LangGraph runs with `stream_mode=["messages", "updates"]`; `api/streaming.py` turns that into typed events:
`token`, `tool_start` (with the search query), `tool_end`, `citations`, `ticket_draft`, `ticket_created`,
`handoff`, `stored`, `done`, `error`. The same union is typed in `frontend/lib/types.ts`.

- **Citation fixing on a live stream:** gpt-oss's `【2†L1-L3】` can be split across chunks, so text from an open `【` is
  held back until its `】` arrives, then rewritten to `[2]`.
- **Structured tool results:** tools return `ToolMessage.artifact` (ticket number, category, priority, duplicate),
  so the API emits `ticket_created` without parsing text.
- **Human mode:** after escalation, customer messages skip the agent and are stored on the ticket. The widget polls
  `/messages?after=<id>` for staff replies; resolving the ticket hands the chat back to the assistant.
- **Guards:** a new message is refused (409) while a ticket draft awaits approve/edit/cancel; resume without a pending
  draft is refused too.

## Live measurements (laptop CPU, Groq gpt-oss-120b free tier)
| Step | Time |
|---|---|
| Doc question: search starts | 1.1 s |
| Retrieval (query embedding + hybrid search) | 0.4 s |
| First answer token | **2.1 s** |
| Full answer (133 streamed chunks) | **2.6 s** |
| Ticket draft shown | 1.2 s |
| Ticket created after approval (DistilBERT + duplicate check) | **0.2 s** (36.5 s before warm-up) |
| AI-suggested staff reply | 2.1 s |

**Startup warm-up** loads DistilBERT and the embedding models once, removing the 30-second first-ticket delay.

## Security notes
- Customer chat endpoints are authorised by the session id (a random UUID). Ticket lookup needs number + email and
  returns the same 404 for "doesn't exist" and "not yours", so ticket numbers can't be probed.
- The customer ticket view hides internal events, classifier confidence and AI messages.
- Staff endpoints require a JWT. A single demo staff account comes from settings; change `JWT_SECRET` and the
  password outside local development.

## Tests
7 API tests (35 in the suite) run the real app, graph and database with a scripted LLM and parse the actual SSE
streams: streamed answer with citations, draft → edit → create, escalation → human mode → staff take/reply → polling
→ resolve back to AI, contact form and scoped lookup, reopen on customer reply, staff auth, filters, reclassification
log, metrics, and the grounded suggested reply.

## Fixed during live testing
- Suggested reply opened with "Hi you," (the prompt said "address the customer as *you*"). It now uses the first name.
- Some page titles contained HTML line breaks. They are cleaned in the loader and at query time (no re-index needed).
