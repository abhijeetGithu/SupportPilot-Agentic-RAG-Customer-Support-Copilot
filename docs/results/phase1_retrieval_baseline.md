# Phase 1: Data, ingestion and dense retrieval baseline

## Data
| Step | Count |
|---|---|
| Article pages in sitemap (`/Content/`, images and archives excluded) | 457 |
| Dropped: thin navigation pages (< 80 words) | 32 |
| Dropped: exact duplicates | 0 |
| **Pages indexed** | **425** (~297K words) |
| **Chunks** (384 tokens, 48 overlap) | **1,563** (avg 267 tokens) |

Cleaning: only the article body (`#mc-main-content`) is kept, converted to markdown so headings survive.
Screenshots are replaced by their captions or alt text (`[Figure: …]`).

## Chunking
1. LlamaIndex `MarkdownNodeParser`: one node per heading section.
2. Small consecutive sections are packed together up to the size limit; oversized sections are split with `SentenceSplitter`.
3. **Contextual chunk header**: `Page title > Section > Heading path` is prepended before embedding,
   so near-identical connector guides stay distinguishable.

Each chunk gets a dense vector (`BAAI/bge-small-en-v1.5`, 384-d) and a BM25 sparse vector (used in Phase 2)
in one Qdrant collection.

## Golden evaluation set
- Stratified sample of 117 chunks across 24 sections (sqrt-proportional, at most 5 from release notes).
- `openai/gpt-oss-120b` (Groq) wrote one paraphrased support-chat question per passage, plus a reference answer and type.
  The LLM could skip unusable passages. Duplicate questions for one passage were removed, leaving **116 questions**.
- Types: 64 how-to, 25 conceptual, 20 factual, 7 troubleshooting.
- **Label expansion:** the same boilerplate (e.g. "Find and Replace", crawl-frequency notes) appears verbatim on 40–56 connector pages.
  For generic questions that don't name a connector, every page holding a near-identical passage
  (word 5-shingle Jaccard ≥ 0.7) also counts as relevant. This applies to 4 of 116 questions. The strict single-page score is reported alongside.
- Status: **v0, awaiting human review** (`evals/datasets/golden_candidates.csv`).

## Results: dense retrieval (bge-small, cosine)
| Metric | Value |
|---|---|
| Recall@1 | 0.776 |
| Recall@3 | 0.897 |
| **Recall@5** | **0.922** |
| Recall@10 | 0.931 |
| MRR@10 | 0.836 |
| Strict Recall@5 (original single label) | 0.888 |
| Latency p50 / p95 (query embed + search, CPU) | ~64 ms / ~105 ms |

### Chunk-size ablation (dense, same golden set)
| Chunk size | Chunks | R@5 | MRR@10 |
|---|---|---|---|
| 256 | 2,394 | 0.905 | 0.798 |
| **384** | 1,563 | **0.922** | **0.836** |
| 512 | 1,237 | 0.871 | 0.772 |

Caveat: questions were generated from the 384-token chunks, which slightly favors that setting.

## Failure analysis (13 strict misses at @5)
| Cause | Count | Fix |
|---|---|---|
| Boilerplate shared across connector pages (the right text, a different page) | 4 | Label expansion (above); collapse duplicate chunks at query time (Phase 2) |
| Release notes duplicate a feature guide; the guide was retrieved instead | 2 | Acceptable: the guide is the better answer. Down-weight release notes (Phase 2) |
| Genuine misses: FAQ wording ("longest search query" vs "maximum characters"), role permissions, settings pages | ~7 | Hybrid search with BM25 for exact terms + reranking (Phase 2) |

## Known limitations
- Questions generated from a single passage are easier than real user queries (median 62% word overlap with
  the source passage). Real-world recall will be lower. A harder set of short, vague, real-style queries is planned.
- 37% of pages rely on screenshots; only captions/alt text are indexed.
