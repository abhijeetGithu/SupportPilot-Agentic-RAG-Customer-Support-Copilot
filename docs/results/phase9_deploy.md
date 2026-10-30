# Phase 9: Free-tier deployment (Render + Vercel + Qdrant Cloud + Neon)

Goal: a public demo at no cost and without a credit card. The constraint that shaped everything: the
only card-free backend host that stays up (Render free) gives **512 MB of RAM**. Hugging Face Docker
Spaces now need a paid plan; Koyeb, Railway and Fly.io have no permanent free compute; Oracle Cloud and
Google Cloud Run are free but need a card. Step-by-step guide: [../DEPLOYMENT.md](../DEPLOYMENT.md).

## Fitting the backend in 512 MB
Measured on the production dependency set (`pip install .`) with a realistic workload: startup + model
warm-up, a real chat answer, a contact-form ticket (classifier + incident detection), a screenshot
upload, and the staff pages (doc gaps, KB search, metrics).

| Step | Peak RSS |
|---|---|
| Start: PyTorch DistilBERT ×2 replaced by int8 ONNX, otherwise unchanged | 529 MB (over the limit) |
| + onnxruntime memory arenas off, llama-index imported only when publishing, classifiers loaded per prediction | 401 MB |
| + fix: the model name was read through the cache, keeping the queue model loaded (caught by a test) | **381 MB** |

Breakdown before the changes (MB): Python + FastAPI + LangChain/LangGraph + app ~235 (qdrant-client 70,
llama-index 34), bge-small embeddings 125, each classifier ~75–99.

## Classifiers: PyTorch → ONNX int8 (`scripts/export_onnx.py`, test split of 2,366 tickets)
| Task | PyTorch macro-F1 | ONNX fp32 | **ONNX int8 (QUInt8)** | Agreement with PyTorch | Size | CPU / ticket |
|---|---|---|---|---|---|---|
| Queue (10 classes) | 0.652 | 0.652 | **0.639** | 87.0% | 268 → **67 MB** | 143 → **96 ms** |
| Priority (3 classes) | 0.716 | 0.716 | **0.704** | 91.6% | 268 → **67 MB** | 142 → **98 ms** |

- The fp32 export matches PyTorch (99.9% identical predictions), so the loss is quantization alone.
- Quantization variants tried on the queue model: signed int8 (default) 0.628; **unsigned int8 0.639
  (chosen)**; keeping the classification head in fp32 changed nothing (the default dynamic quantization
  already leaves those Gemm nodes in fp32); per-channel variants broke the model (macro-F1 ~0.01).
- Keeping the 94 MB fp32 embedding table would recover more accuracy but doesn't fit the memory budget.
- Production loads one model per prediction (~1 s extra per new ticket) and releases it.

## Dependencies: the reason CI was failing
GitHub Actions had failed on every push since Phase 7.5: the incident detector imported `scipy` and the
doc-gap clustering `scikit-learn`, both installed only on the development machine, so the deployed
backend would also have crashed on those pages. Replaced by small NumPy/stdlib versions, tested against
the reference libraries (Poisson tail to 1e-12; clustering identical, ARI 1.0 on 5 random sets).
Runtime dependencies trimmed (optional LLM providers, crawler and tracing moved to extras). A clean
environment installed exactly like CI passes all 103 tests.

## Public-demo safeguards
- **Rate limits** per client IP (chat, sessions, uploads, tickets, lookups, login) and a **daily cap**
  on assistant turns (300) to protect the Groq free tier. The client IP is the last `X-Forwarded-For`
  entry (added by Render's proxy): the first test version trusted the first entry, which a client can
  forge; a test now covers it.
- **Read-only demo login** for visitors: every non-GET staff request is refused (except compute-only
  "suggest reply"/"classify"). The admin login stays private.
- **Production mode** refuses to start with default secrets, a weak staff password, no Groq key or SQLite.
- Constant-time password comparison; staff tokens carry a role.
- `.env.example` had end-of-line comments after empty values: some parsers read the comment as the
  value, which would have enabled the demo login with a guessable password. Comments moved to own lines.

## Frontend
- The browser calls the Render API directly (`NEXT_PUBLIC_API_BASE`): a Vercel rewrite proxy can buffer
  the SSE token stream and adds timeouts.
- A "Waking up the AI backend" bar while the free instance starts (only shown if `/health` is slow).
- The login page shows the read-only demo credentials; the dashboard marks the demo account.

## Deployment files
`Dockerfile` (slim, bakes the embedding models, downloads the classifiers from a GitHub release),
`render.yaml` (Render Blueprint), `scripts/migrate_qdrant.py` (copies vectors + payloads to Qdrant Cloud,
refuses to overwrite its own source; tested: identical search results on the copy),
`scripts/package_models.py`, `docs/DEPLOYMENT.md`.

## Not done yet
- The live deployment itself: it needs free accounts (Qdrant Cloud, Neon, Render, Vercel) created by
  the owner, following the guide.
- Langfuse tracing and an evaluation gate in CI.
