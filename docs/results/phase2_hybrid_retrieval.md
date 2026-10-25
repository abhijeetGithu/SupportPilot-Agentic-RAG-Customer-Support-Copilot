# Phase 2: Hybrid search, duplicate collapsing, reranking and query rewriting

## Two test sets
| Set | Questions | How it was made | Word overlap with answer passage |
|---|---|---|---|
| **Standard** | 116 | LLM-written support questions (Phase 1) | 0.62 |
| **Hard** (vocabulary mismatch) | 116 | Same questions rewritten as real users type: keywords, symptom descriptions, casual with typos, everyday synonyms. Doc terms were explicitly banned per question; typos injected in code (seeded) | **0.20** |

Labels are shared, so results compare directly. The hard set is a stress test. Some rewrites drop
information a real user would include (e.g. "adjust display box dimensions limits" without naming Agent Helper),
so treat its scores as a lower bound.

A first version of the hard set (v1, `golden_hard_v1.jsonl`) asked the LLM to avoid doc wording in general terms.
It ignored the instruction (overlap stayed at 0.67, 2 of 29 typo queries had typos), which is why v2 names the banned words.

## Results (Recall@5, 116 questions each; 1 question = 0.86 points)

| Strategy | Standard R@5 | Standard MRR | Hard R@5 | Hard MRR | Latency p50 |
|---|---|---|---|---|---|
| Dense (Phase 1 baseline) | 0.922 | 0.836 | 0.543 | 0.403 | ~80 ms |
| BM25 only | 0.922 | 0.780 | 0.310 | 0.216 | ~55 ms |
| Hybrid, RRF k=2 + dedup | **0.966** | 0.853 | 0.509 | 0.378 | ~75 ms |
| + MiniLM-L6 reranker | 0.948 | 0.859 | 0.491 | 0.403 | ~2,200 ms |
| + Jina-tiny reranker | 0.948 | 0.879 | 0.474 | 0.340 | ~1,900 ms |
| + Jina-turbo reranker | 0.957 | 0.864 | 0.483 | 0.386 | ~2,600 ms |
| Dense + LLM query rewrite | 0.914 | 0.826 | 0.612 | 0.470 | ~85 ms + LLM |
| **Hybrid + dedup + LLM query rewrite (final)** | **0.966** | **0.857** | **0.638** | **0.431** | ~90 ms + ~1 s LLM |

**Final vs Phase 1 baseline:** standard 0.922 → **0.966** (misses 9 → 4); hard 0.543 → **0.638** (misses 53 → 42).

Latency was measured on a laptop CPU (Ryzen 5 5600H) **on battery**, so absolute numbers are pessimistic.

## What each component did

**Hybrid search (dense + BM25, weighted RRF).** +4.4 points on the standard set. Dense and BM25 fail on *different*
questions: BM25 catches exact terms like error codes, `getCaseDeflectedArticles` and connector names. On the hard set
alone, hybrid is *worse* than dense (0.509 vs 0.543): with no shared vocabulary, BM25 only adds noise. Down-weighting
BM25 (0.5, 0.3) did not fix this; only weight 0 (pure dense) did.

**RRF constant k.** Swept 2 / 10 / 20 / 60. The textbook k=60 gave 0.948 on the standard set and k=2 gave 0.966.
A small k rewards documents ranked at the very top of either list, which suits a small, precise corpus. Default: k=2.

**Duplicate collapsing.** 249 chunks (16%) sit in 20 near-duplicate groups (word 5-shingle Jaccard ≥ 0.8). The largest is
"After the First Crawl", copied onto 56 connector pages. Collapsing keeps the best-ranked copy only. Recall is unchanged
because the labels already accept duplicates, but the top 5 now holds 5 *different* passages for the LLM instead of
several copies of one paragraph.

**Cross-encoder reranking: rejected.** Three rerankers were tested (MiniLM-L6, Jina-tiny, Jina-turbo). None improved Recall@5
on either set, while each added 1.9–2.6 s per query on CPU. Jina-tiny improves R@1/MRR on the standard set (0.767 → 0.819)
but loses top-5 recall. The larger BGE rerankers took 12–27 s per query, which is unusable for chat. MiniLM is trained on
web search (MS MARCO), not product docs. The reranker stays in the code, switched off (`USE_RERANKER=false`).

**LLM query rewriting: adopted.** The LLM (gpt-oss-120b on Groq, with a 15-line product glossary) rewrites the user's
question into doc vocabulary, e.g. "the AI link so tools like the chatbot are inactive" → "disable … LLM integration
SearchUnifyGPT". Both the original and the rewritten query are searched and fused, so a bad rewrite cannot remove
good matches. On the hard set this gives **+12.9 points** over hybrid (0.509 → 0.638) and turns BM25 useful again,
because the rewrite contains the doc's own terms. On the standard set it is neutral (0.966 → 0.966). Cost: about 1 s
and one small LLM call per query. In the agent (Phase 4) the LLM already writes the `search_docs` query, so this may
merge into that step.

**Release-notes down-weighting: no effect.** Release notes rank #1 in 7 of the 42 remaining hard-set misses, because they
mention many features briefly. A score penalty (×0.5, ×0.25, skipped when the query asks about releases) left
Recall@5 unchanged and moved MRR within noise. Kept off (`RELEASE_NOTES_WEIGHT=1.0`).

## Remaining failures (hard set, 42)
- Queries missing the key subject (a generation artefact; a real user would name "Agent Helper").
- Conceptual "how does SearchUnify work" questions whose answer sits in `Concepts/` or `FAQs/` but whose wording
  matches feature pages (NLP Manager, NER).
- Release notes outranking the feature guide.

## Not yet verified
- The chunk-size choice (384) was tuned on the standard set only. Re-checking it on the hard set needs two re-indexes (~15 min).
- Both sets are v0 (not yet human-reviewed). Numbers may shift after review.

## Production defaults (`config.py`)
`hybrid_sparse_weight=1.0`, `rrf_k=2`, dedup on, `use_reranker=False`, `query_rewrite=True`, `release_notes_weight=1.0`.
