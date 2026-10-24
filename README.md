<div align="center">

# 🛟 SupportPilot

### Agentic RAG Customer-Support Copilot

An AI support platform that answers product questions from documentation, creates and tracks
tickets, routes them with a fine-tuned classifier, and hands off to human agents.
It is evaluated, guarded and traced like a production system.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-8A2BE2)
![Qdrant](https://img.shields.io/badge/Qdrant-hybrid%20search-DC244C)
![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-orange)

</div>

---

## 📌 Overview

Support teams answer the same documentation questions over and over, and tickets arrive
without category or priority. **SupportPilot** puts an AI agent in front of the support queue:

- **Customers** chat with an assistant that answers from the product docs, **cites its sources**,
  and creates or checks support tickets when the docs aren't enough.
- **Staff** get a dashboard where tickets are already **classified by category and priority**,
  with **AI-drafted replies** they review before sending, and a queue for live handoffs.
- **Developers** can use the same tools from Claude Desktop or Cursor through an **MCP server**.

The knowledge base is the public **[SearchUnify documentation](https://docs.searchunify.com/Content/Home.htm)**
(enterprise search and support platform), used with permission from Grazitti.

---

## ✨ Key features

| Area | Feature | Status |
|---|---|---|
| **RAG** | Doc crawler, HTML cleaning, heading-aware chunking with contextual headers | ✅ Done |
| | Dense retrieval (bge-small) + BM25 vectors in Qdrant | ✅ Done |
| | Hybrid search (Reciprocal Rank Fusion) + cross-encoder reranking | 🔄 In progress |
| **Evaluation** | 116-question golden set, Recall@K / MRR, failure analysis, chunk-size ablation | ✅ Done |
| | RAGAS faithfulness / relevancy, LLM-as-judge, agent scenario tests, CI quality gate | 📋 Planned |
| **Agent** | LangGraph agent with tool calling, memory and step limits | 📋 Planned |
| | Ticket creation with **human-in-the-loop confirmation** (LangGraph `interrupt`) | 📋 Planned |
| | Escalation to a human with **live handoff** in the chat | 📋 Planned |
| **ML** | DistilBERT ticket classifier vs TF-IDF baseline, tracked in MLflow, staff-feedback loop | 📋 Planned |
| **App** | FastAPI backend with SSE streaming, JWT staff auth | 🏗️ Scaffolded |
| | Next.js customer chat widget, support pages and staff dashboard | 🏗️ Scaffolded |
| | PostgreSQL ticket system (sessions, tickets, messages, audit events) | 🏗️ Scaffolded |
| **Integrations** | MCP server exposing `search_docs`, `create_ticket`, `get_ticket_status` | 📋 Planned |
| **LLMOps** | Prompt-injection and PII guardrails, Langfuse tracing (latency, tokens, cost) | 📋 Planned |
| **Deploy** | Docker Compose, GitHub Actions CI, Vercel + Render deployment | 🏗️ Scaffolded |

✅ Done · 🔄 In progress · 🏗️ Scaffolded (structure and interfaces in place) · 📋 Planned

---

## 🏗️ System architecture

```mermaid
flowchart LR
    subgraph Clients
        CU["👤 Customer"]
        ST["🧑‍💼 Support staff"]
        DEV["🤖 Claude Desktop / Cursor"]
    end

    subgraph FE["Frontend · Next.js + TypeScript"]
        CW["Chat widget"]
        SP["Support pages<br/>contact form · ticket status"]
        DB_UI["Staff dashboard<br/>inbox · escalations · metrics"]
    end

    subgraph BE["Backend · FastAPI"]
        API["REST + SSE API"]
        GIN["Input guardrails<br/>prompt injection · PII"]
        AG["LangGraph agent"]
        GOUT["Output guardrails<br/>grounding · PII redaction"]
        TS["TicketService<br/>dedupe · classify · audit"]
        MCP["MCP server"]
    end

    subgraph AI["AI services"]
        RET["Hybrid retriever<br/>dense + BM25 → RRF → rerank"]
        CLF["DistilBERT<br/>ticket classifier"]
        LLM["LLM · Groq<br/>gpt-oss-120b"]
    end

    subgraph DATA["Data stores"]
        QD[("Qdrant<br/>1,563 doc chunks")]
        PG[("PostgreSQL<br/>tickets · sessions")]
    end

    LF["📊 Langfuse<br/>traces · cost · latency"]

    CU --> CW & SP
    ST --> DB_UI
    CW & SP & DB_UI -->|"/api via Next.js proxy"| API
    DEV -->|stdio| MCP

    API --> GIN --> AG --> GOUT --> API
    AG -->|tool calls| RET
    AG -->|tool calls| TS
    AG --> LLM
    API --> TS
    MCP --> RET & TS

    RET --> QD
    TS --> CLF
    TS --> PG
    AG -.-> LF
```

**Design principles**
- **One write path for tickets:** chat, contact form, staff dashboard and MCP all go through `TicketService`,
  so classification, duplicate detection and the audit log are applied the same way everywhere.
- **The LLM never chooses identity:** the customer's email comes from the chat session (LangGraph `InjectedState`),
  so a prompt-injected agent cannot read or create tickets for someone else.
- **Every quality claim is measured:** retrieval, answers, classifier and agent behaviour each have an eval with stored reports.

---

## 🔄 Application flow

### 1. Customer asks a question (RAG answer)

```mermaid
sequenceDiagram
    autonumber
    actor C as Customer
    participant W as Chat widget
    participant API as FastAPI
    participant G as Guardrails
    participant A as LangGraph agent
    participant R as Hybrid retriever
    participant Q as Qdrant
    participant L as LLM

    C->>W: "How do I connect Jira so SearchUnify indexes our issues?"
    W->>API: POST /api/chat/sid/stream
    API->>G: check input for prompt injection and PII
    G-->>API: allowed, PII masked
    API->>A: run with message and session customer
    A->>L: decide next step with tools available
    L-->>A: call search_docs
    A->>R: search_docs query
    R->>Q: dense + BM25 search, top 20
    Q-->>R: candidate chunks
    R-->>A: top 5 after RRF fusion and reranking
    A->>L: answer using only these passages, cite sources
    L-->>A: grounded answer with citations
    A->>G: output check for grounding and PII
    API-->>W: SSE stream of tokens, citations, done
    W-->>C: Answer with clickable source links
```

### 2. Docs can't solve it, so a ticket is created (human in the loop)

```mermaid
sequenceDiagram
    autonumber
    actor C as Customer
    participant W as Chat widget
    participant API as FastAPI
    participant A as LangGraph agent
    participant TS as TicketService
    participant CLF as Classifier
    participant DB as PostgreSQL

    C->>W: "Our Jira crawl keeps failing with a 401 error"
    W->>API: POST /api/chat/sid/stream
    API->>A: run
    A->>A: search_docs finds no fix
    A->>A: create_ticket drafts subject and description
    A-->>API: interrupt with ticket draft
    API-->>W: SSE ticket_draft
    W-->>C: Editable confirmation card
    C->>W: Reviews, edits, clicks Create ticket
    W->>API: POST /api/chat/sid/resume approve
    API->>A: resume graph
    A->>TS: create ticket for the session's customer
    TS->>TS: mask PII, check for a duplicate in the last 24h
    TS->>CLF: classify subject and description
    CLF-->>TS: Technical · High · 0.91
    TS->>DB: insert ticket, transcript, audit events
    TS-->>A: SP-01042
    API-->>W: SSE ticket_created
    W-->>C: Ticket SP-01042 created · Technical · High
```

### 3. Escalation and live handoff to a human

```mermaid
sequenceDiagram
    autonumber
    actor C as Customer
    participant W as Chat widget
    participant A as Agent
    participant TS as TicketService
    participant D as Staff dashboard
    actor S as Support agent

    C->>A: "This is urgent, I need to talk to a person"
    A->>TS: escalate_to_human with reason
    TS->>TS: ticket status escalated, session mode human
    A-->>W: SSE handoff
    W-->>C: Connecting you to the support team
    D->>S: Ticket appears in the escalation queue
    S->>D: Take ticket
    S->>D: Ask for an AI-suggested reply
    D-->>S: RAG-drafted reply with citations
    S->>D: Edit and send
    W->>W: Polls for new messages in human mode
    W-->>C: Staff reply shown live in the chat
    S->>D: Mark resolved, session returns to AI mode
```

### 4. Ticket lifecycle

```mermaid
stateDiagram-v2
    [*] --> open: created via chat, form or MCP
    open --> in_progress: staff picks up
    open --> escalated: agent or customer requests a human
    escalated --> in_progress: staff takes the escalation
    in_progress --> resolved: staff replies with a fix
    resolved --> in_progress: customer replies again
    resolved --> closed: auto-close or confirmed
    closed --> [*]
```

Tickets with classifier confidence below 0.6 are flagged `needs_review`. When staff change a category
or priority, a `reclassified` event is logged and exported as new training data for the classifier.

---

## 🧠 Agent design (LangGraph)

```mermaid
flowchart TD
    S([Start]) --> GI[guard_input]
    GI -->|blocked| GO
    GI --> AG[agent · LLM with tools]
    AG -->|tool calls| T[tools]
    T --> AG
    AG -->|final answer or step limit| GO[guard_output]
    GO --> E([End])

    T -.-> T1["search_docs"]
    T -.-> T2["create_ticket<br/>⏸ interrupt for confirmation"]
    T -.-> T3["get_ticket_status<br/>scoped to session email"]
    T -.-> T4["escalate_to_human<br/>ends the turn"]
```

| Tool | Purpose |
|---|---|
| `search_docs(query)` | Hybrid search + rerank, returns passages with source URLs |
| `create_ticket(subject, description)` | Drafts a ticket and **pauses** until the customer confirms or edits it |
| `get_ticket_status(number)` | Returns status and latest reply, only for the session's own tickets |
| `escalate_to_human(reason)` | Escalates the ticket and switches the chat to human mode |

Conversation memory is a LangGraph checkpointer keyed by chat session.

---

## 📚 RAG pipeline

### Offline ingestion

```mermaid
flowchart LR
    A["🌐 docs.searchunify.com<br/>sitemap · robots.txt"] --> B["Crawler<br/>457 pages"]
    B --> C["Cleaner<br/>article body only<br/>screenshots → captions<br/>HTML → markdown"]
    C --> D["Filter<br/>drop 32 thin pages<br/>425 kept"]
    D --> E["Chunker · LlamaIndex<br/>split by headings<br/>pack small sections<br/>384 tokens"]
    E --> F["Contextual header<br/>Title › Section › Heading"]
    F --> G1["Dense embedding<br/>bge-small-en-v1.5"]
    F --> G2["Sparse embedding<br/>BM25"]
    G1 & G2 --> H[("Qdrant<br/>1,563 chunks")]
```

**Why the contextual header?** SearchUnify has 140 nearly identical connector guides ("Use Jira…",
"Use Confluence…"). Prepending `Page title › Section › Heading` before embedding keeps them distinguishable.

### Query time

```mermaid
flowchart LR
    Q["User question"] --> E1["Embed query"]
    E1 --> D["Dense search<br/>top 20"]
    E1 --> S["BM25 search<br/>top 20"]
    D & S --> F["Reciprocal Rank Fusion"]
    F --> X["Collapse duplicate<br/>boilerplate chunks"]
    X --> R["Cross-encoder rerank<br/>ms-marco-MiniLM"]
    R --> T["Top 5 passages<br/>with citations"]
```

---

## 📊 Evaluation

```mermaid
flowchart LR
    CH["Indexed chunks"] --> SMP["Stratified sample<br/>24 doc sections"]
    SMP --> GEN["LLM writes realistic<br/>support questions"]
    GEN --> REV["Human review<br/>keep · fix · drop"]
    REV --> EXP["Label expansion<br/>near-duplicate pages"]
    EXP --> GS[("Golden set<br/>116 questions")]
    GS --> RE["Retrieval eval<br/>Recall@K · MRR"]
    GS --> AE["Answer eval<br/>RAGAS · LLM-as-judge"]
    RE & AE --> REP["Reports +<br/>failure analysis"]
    REP --> CI["CI gate<br/>fail if quality drops"]
```

### Results so far

**Dense retrieval baseline** (116 questions, bge-small, 384-token chunks):

| Metric | Score |
|---|---|
| Recall@1 | 0.776 |
| Recall@3 | 0.897 |
| **Recall@5** | **0.922** |
| Recall@10 | 0.931 |
| MRR@10 | 0.836 |
| Search latency p50 / p95 (CPU) | ~64 ms / ~105 ms |

**Chunk-size ablation**

| Chunk size | Chunks | Recall@5 | MRR@10 |
|---|---|---|---|
| 256 | 2,394 | 0.905 | 0.798 |
| **384** | 1,563 | **0.922** | **0.836** |
| 512 | 1,237 | 0.871 | 0.772 |

**Failure analysis:** 4 of 13 misses were boilerplate text copied across 40–56 connector pages
(the right answer, from a different page). These were fixed with near-duplicate label expansion.
The rest were exact-term mismatches, which Phase 2 targets with hybrid search and reranking.
Full write-up: [docs/results/phase1_retrieval_baseline.md](docs/results/phase1_retrieval_baseline.md).

| Upcoming metric | Phase |
|---|---|
| Recall@5 with hybrid + rerank, plus a harder real-user query set | 2 |
| Ticket classifier macro-F1 (DistilBERT vs TF-IDF baseline) | 3 |
| RAGAS faithfulness and answer relevancy, agent tool-choice accuracy | 8 |
| p95 end-to-end latency and cost per query | 9 |

---

## 🧰 Tech stack

| Layer | Technologies |
|---|---|
| **LLM** | Groq (`openai/gpt-oss-120b`), switchable to Gemini or Ollama via one env var |
| **Agent** | LangGraph (state graph, tools, checkpointer, `interrupt` for human in the loop) |
| **RAG** | LlamaIndex (parsing, chunking), Qdrant (dense + sparse vectors), fastembed (bge-small, BM25), cross-encoder reranker |
| **ML** | DistilBERT (Hugging Face Transformers), scikit-learn baseline, MLflow |
| **Evaluation** | Custom Recall@K / MRR harness, RAGAS, LLM-as-judge |
| **Backend** | FastAPI, Server-Sent Events, SQLAlchemy, PostgreSQL, JWT |
| **Frontend** | Next.js 16 (App Router), TypeScript, Tailwind CSS, shadcn/ui, TanStack Query |
| **Integrations** | Model Context Protocol (official Python SDK) |
| **LLMOps** | Langfuse tracing, guardrails (Llama Prompt Guard, PII masking) |
| **DevOps** | Docker Compose, GitHub Actions, Vercel, Render |

---

## 📁 Project structure

```
├── src/supportpilot/
│   ├── config.py            # settings from .env
│   ├── llm/                 # LLM provider switch (Groq / Gemini / Ollama)
│   ├── ingestion/           # HTML loader, cleaning, chunking, indexing pipeline
│   ├── retrieval/           # embeddings, Qdrant store, dense / hybrid search, reranker
│   ├── agent/               # LangGraph state, tools, prompts, graph
│   ├── tickets/             # TicketService (create, dedupe, classify, audit)
│   ├── db/                  # SQLAlchemy models: sessions, tickets, messages, events
│   ├── classifier/          # TF-IDF baseline, DistilBERT training and inference
│   ├── guardrails/          # input / output checks
│   ├── evaluation/          # retrieval metrics
│   ├── observability/       # Langfuse
│   ├── api/                 # FastAPI app, routes (chat, tickets, admin, auth)
│   └── mcp_server/          # MCP server
├── frontend/                # Next.js app (chat widget, support pages, staff dashboard)
├── evals/                   # golden set generation, label expansion, eval runners
├── scripts/                 # crawl_docs.py, ingest.py
├── tests/                   # pytest unit tests
├── docs/                    # ARCHITECTURE.md, results per phase
├── docker-compose.yml       # postgres · qdrant · api · web
└── .github/workflows/ci.yml # lint, tests, frontend build
```

Detailed design (API endpoints, database schema, SSE events, security): **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

## 🚀 Getting started

**Prerequisites:** Python 3.11+, Node.js 20+, Docker Desktop, and a free [Groq API key](https://console.groq.com).

```bash
# 1. Backend setup
python -m venv .venv
.venv\Scripts\activate                 # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
cp .env.example .env                   # add your GROQ_API_KEY

# 2. Start Qdrant and PostgreSQL
docker compose up -d qdrant postgres

# 3. Build the knowledge base
python scripts/crawl_docs.py           # download the docs (~18 MB)
python scripts/ingest.py --recreate    # clean, chunk, embed, index into Qdrant

# 4. Evaluate retrieval
python evals/run_retrieval_eval.py --strategy dense

# 5. Run the API and the frontend
uvicorn supportpilot.api.main:app --reload       # http://localhost:8000/docs
cd frontend && npm install && npm run dev        # http://localhost:3000
```

Run the tests with `pytest`. To run everything in containers, use `docker compose up --build`.

---

## 🗺️ Roadmap

- [x] **Phase 1:** Data crawling, cleaning, chunking, dense retrieval baseline, golden set, retrieval eval
- [ ] **Phase 2:** Hybrid search (BM25 + dense, RRF), cross-encoder reranking, harder query set
- [ ] **Phase 3:** Ticket classifier: TF-IDF baseline vs fine-tuned DistilBERT, MLflow tracking
- [ ] **Phase 4:** LangGraph agent, tools, ticket confirmation interrupt, TicketService, memory
- [ ] **Phase 5:** FastAPI chat (SSE), tickets, admin and auth endpoints
- [ ] **Phase 6:** Next.js chat widget, support pages, staff dashboard
- [ ] **Phase 7:** MCP server for Claude Desktop / Cursor
- [ ] **Phase 8:** RAGAS and LLM-as-judge evals, agent scenario tests, guardrails
- [ ] **Phase 9:** Langfuse observability, CI eval gate, cloud deployment and live demo

---

## 📄 Data and permissions

The knowledge base is the public SearchUnify documentation at docs.searchunify.com, **used with permission
from Grazitti** for this non-commercial portfolio project. The crawler respects `robots.txt` and rate-limits
its requests. Raw pages are not committed to this repository; they can be re-downloaded with `scripts/crawl_docs.py`.

---

## 👤 Author

**Abhijeet Singh**: [LinkedIn](https://www.linkedin.com/in/your-profile) · [Email](mailto:your-email@example.com)
