# Phase 6: Next.js frontend

**Stack:** Next.js 16 (App Router, Cache Components, Turbopack) · React 19 · TypeScript · Tailwind CSS 4 ·
shadcn/ui (Base UI primitives) · TanStack Query · react-markdown · Recharts.

## Pages
| Route | Who | What it does |
|---|---|---|
| `/` | Customer | Landing page, popular questions that open the chat, floating chat widget |
| `/support/new` | Customer | Contact form → ticket (auto-classified, duplicates merged) |
| `/support/tickets?number=&email=` | Customer | Ticket status, staff replies, add a reply (reopens a resolved ticket) |
| `/login` | Staff | JWT sign-in |
| `/dashboard` | Staff | Stat tiles + tickets by category / priority / status |
| `/dashboard/tickets` | Staff | Inbox: search, filters (status, priority, category, needs review), urgent-first, pagination |
| `/dashboard/tickets/[id]` | Staff | Conversation, classification controls with confidence, **Suggest reply**, audit timeline |
| `/dashboard/escalations` | Staff | Customers waiting for a person; "Take conversation" |

## Chat widget
- Pre-chat form (name, email), then a session that survives page reloads (`GET /chat/{sid}/history`).
- Streams the answer token by token over SSE (`fetch` + `ReadableStream`, since `EventSource` only supports GET),
  with a live status line ("Searching the docs for 'jira content source setup'…").
- `[n]` citations become links to the doc page; sources are grouped by page (`[1][2] Use Jira Software…`).
- **Ticket draft card:** the agent pauses, and the customer edits the subject or description, then creates or cancels.
  The input is locked while a draft is pending (the API refuses new messages with 409).
- **Handoff mode:** after escalation the header switches to "Connected to the support team"; the widget polls for
  staff replies every 4 s and returns to the assistant when the ticket is resolved.

## Verification
- `tsc`, ESLint (incl. React Compiler rules) and `next build` all clean; all pages prerender (ticket detail: partial
  prerender).
- **SSE through the Next.js proxy:** 118 tokens arrived at 26 distinct moments (not buffered); first token 1.5 s.
- **Browser walk-through (Playwright, real backend, Groq, DistilBERT):** customer asks a doc question → answer with
  sources → asks for a ticket → edits/approves the draft → asks for a person → staff signs in → takes the escalation →
  "Suggest reply" drafts a cited answer → staff sends → the reply appears in the customer's chat (~1–3 s, polling).
  **0 console or page errors.** All 13 steps took ~50 s on a laptop.

## Fixed after looking at the screenshots
| Problem | Fix |
|---|---|
| Sources listed the same page 4 times (several chunks from one page) | Chips grouped by page: `[1][2][3] Title` |
| Ticket draft written in the third person ("Customer reports that…") although the customer approves it | Agent instructed to write in the customer's voice ("Since this morning our Zendesk search client…") |
| Raw markdown (`**bold**`) in the staff transcript and in AI-drafted replies shown to customers | Markdown rendering for AI and staff messages everywhere |
| Charts with a single category would be one-bar charts | A sentence instead ("The only ticket so far: …") |

## Charts
Built with the dataviz guidelines: single series → one validated hue (`#2a78d6` light / `#3987e5` dark, checked with the
palette validator against the app's own surfaces), thin 18 px bars with a rounded data end, value at each bar tip in text
colour, no gridlines (every value is labelled), per-bar tooltip with the share, "View as table" for every chart, ordinal
scales (priority, status) kept in natural order, and the frame held (dimmed) while refreshing instead of a skeleton.

## Engineering notes
- **Next.js 16 Cache Components:** client hooks that read the URL (`useParams`, `useSearchParams`, `usePathname`) must
  sit inside `<Suspense>`. The production build caught the sidebar's `usePathname` on the dynamic ticket route; the
  active-link nav is now its own Suspense island.
- **React Compiler lint (no setState in effects):** the staff token is read with `useSyncExternalStore` (no hydration
  mismatch), the ticket tracker uses TanStack Query keyed by the URL, and the widget restores its session with a lazy
  state initializer.
- **Windows dev note:** stopping `npm run start` can leave the Node child process holding port 3000; stop it by port.
