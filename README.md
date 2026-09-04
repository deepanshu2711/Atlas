# Atlas

A multi-tenant retrieval service that answers questions over documents nobody chunked nicely — with clickable citations and an eval harness that tells you when a change made it worse.

Project 02 of the [AI Systems Build Lab](.). Core skill: retrieval quality and evaluation discipline.

| | |
|---|---|
| **Language** | Python + Postgres + Next.js |
| **Time to v1** | 3 weeks part-time |
| **Rebuilds** | RAGFlow, LightRAG, R2R, Onyx |

## Why this project

Almost every RAG tutorial ends where the interesting work starts. Split on 512 characters, embed, top-k, stuff into a prompt — that pipeline gets roughly 60% of questions right on real documents, and the remaining 40% are the ones people actually care about: the conditional clause, the number in the table on page 31, the thing that requires connecting two sections.

Retrieval has ground truth, unlike most agent work. You can build an ablation table and watch each technique earn or fail to earn its complexity — measure, then decide.

## The actual problem

Retrieval fails in five distinct ways, and they need five different fixes:

| Failure | Looks like | Fix |
|---|---|---|
| Lost structure | A table becomes a wall of numbers; a clause is severed from its heading | Layout-aware parsing, structure-preserving chunking |
| Orphaned chunk | "This exclusion does not apply if…" — which exclusion? | Contextual retrieval: prepend a situating sentence before embedding |
| Vocabulary mismatch | User says "burst pipe", the document says "escape of water" | Hybrid dense + sparse, query expansion |
| Multi-hop | Answer requires the definition on p.3 and the exception on p.31 | Query decomposition, or a graph edge between them |
| Top-k truncation | Right chunk retrieved at rank 14, prompt only takes 5 | Retrieve wide (40–60), rerank narrow (5–8) |

## Architecture

```
INGEST (async, per tenant)
upload -> S3/MinIO -> job queue
    |
    v
[ parse ]      docling / unstructured -> typed blocks
    |          (heading, para, table, figure, list, page_no, bbox)
    v
[ chunk ]      structure-aware split, respects headings + table atomicity
    |
    v
[ contextualize ] per chunk: LLM writes 1-2 sentences situating it in the doc
    |
    +--> dense index    pgvector HNSW (chunk + context, embedded together)
    +--> sparse index   Postgres tsvector / BM25
    +--> entity graph   extract entities & relations -> Kuzu
    +--> doc summary    used for routing when a tenant has many documents


QUERY (sync)
POST /v1/query {tenant, question, mode}
    |
    v
[ understand ]  rewrite w/ conversation history
    |           classify: factoid | comparative | multi-hop | aggregate | unanswerable
    |           decompose into 1-3 sub-questions if needed
    v
[ route ]       which collections / documents are in scope?
    |
    v
+---- per sub-question ------------------------------+
|  dense k=40   sparse k=40   graph k=20              |
|          \        |        /                        |
|           reciprocal rank fusion                    |
|                   |                                 |
|           cross-encoder rerank -> top 8              |
+----------------------------------------------------+
    |
    v
[ assemble ]   dedupe, expand to parent block, order by document position
    |
    v
[ generate ]   answer w/ mandatory [chunk_id] citations
    |
    v
[ verify ]     every sentence's citation actually supports it? (cheap NLI or LLM judge)
    |          fail -> retry once with "you cited X but X doesn't say that"
    v
{ answer, citations:[{chunk_id, doc, page, bbox, quote}], confidence }
```

**The one design decision that matters most:** retrieve wide, rerank narrow. Dense-only top-5 is the default and it is the single biggest cause of bad answers. Pulling 40 candidates from three retrievers and letting a cross-encoder pick 8 typically moves answer accuracy more than switching to a better LLM — and it costs milliseconds on CPU.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI, async, SSE for streaming | Streaming citations as they resolve is a real UX difference |
| Parsing | docling (IBM) or unstructured | Layout blocks with page + bbox make clickable citations possible |
| Store | Postgres 16 + pgvector + tsvector | One database, one transaction, row-level security for tenancy |
| Graph | Kuzu (embedded Cypher) | Entity/relation index for multi-hop, without operating Neo4j |
| Embeddings | bge-m3 dense + SPLADE or BM25 sparse | bge-m3 does dense, sparse, and multi-vector from one model |
| Reranker | bge-reranker-v2-m3 (CPU) or Jina/Cohere API | Start local; measure whether the API version is worth it |
| Workers | Arq or Celery + Redis | Ingestion is minutes-long and must survive restarts |
| Blob | MinIO (S3 API) | Keep originals — you will re-ingest many times as chunking changes |
| Eval | Ragas + a golden set you own, promptfoo for prompts | Faithfulness, context precision, context recall out of the box |
| Front end | Next.js + shadcn | A citation viewer that highlights the source bbox on the PDF page |
| Tracing | Langfuse (MIT, self-hosted) | Per-query retrieval traces to debug anything |

## Ingestion, properly

- **Never split a table.** One chunk, serialised as markdown, with its caption and nearest heading attached. If it exceeds the size limit, split by rows and repeat the header row.
- **Never orphan a heading.** Every chunk carries its full heading path: `Section 4 > 4.2 Exclusions > 4.2.3 Vacancy`.
- **Split at paragraph boundaries** within a target range (300–900 tokens), not at a fixed size.
- **Keep `page_no` and `bbox` on every chunk.** Without them there are no clickable citations, and clickable citations are what make people trust the system.
- **Contextualize before embedding.** A cheap model writes 1-2 sentences situating each chunk in the document; context + chunk are embedded together. Highest ratio of quality gain to implementation effort in the whole pipeline ([Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)).
- **Extract an entity graph.** Entities + typed relations per chunk, with edges back to the source chunk. Entity→entity and entity→chunk edges are ~90% of what GraphRAG buys you at ~10% of the cost.

## The retrieval ladder

Build these in order, measure after each — do not add the next rung until the current one is in the ablation table:

1. Dense top-k (baseline)
2. + Sparse (BM25) with RRF — fixes identifier/jargon lookups
3. + Cross-encoder rerank over 40 candidates — usually the largest single gain
4. + Contextual chunks (requires re-ingestion; measure separately)
5. + Query decomposition — only helps multi-hop; segment eval by question type
6. + Graph expansion — helps entity-centric/comparative, costs ingest time
7. + Parent-block expansion — retrieve small, return large

## The agentic loop

"Agentic RAG" means the retriever can decide it hasn't found enough and try again, on a hard budget:

```
state = {question, subqs: [], evidence: [], hops: 0, budget: 4}

loop:
  if hops >= budget: break
  gaps = assess(question, evidence)      # LLM: what is still missing? -> list | []
  if not gaps: break
  action = plan(gaps)                    # SEARCH | EXPAND | ZOOM | LOOKUP_TABLE | CLARIFY
  evidence += execute(action)
  hops += 1

answer = generate(question, evidence)
verify(answer, evidence)                  # every claim must map to a chunk_id
```

`assess` must be allowed to return "nothing missing" (and should be rewarded for stopping). If a search hop returns only chunks already in evidence, break rather than let the model rephrase forever.

## Multi-tenancy

This is where the project becomes systems work rather than notebook work — isolation enforced by the database, not by application `WHERE` clauses:

```sql
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
  USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- every request sets it once, in middleware, inside the transaction
SET LOCAL app.tenant_id = '...';
```

| Concern | Approach |
|---|---|
| Data isolation | RLS by `tenant_id`; CI test asserts zero rows on wrong tenant |
| Index isolation | Partial HNSW per large tenant, or partition `chunks` by tenant hash |
| Config per tenant | Embedding model, chunk size, reranker, LLM provider — versioned rows; changing the embedding model triggers re-ingestion, never silent mixing |
| Quotas | Documents, storage, queries/min, tokens/month — enforced in the gateway |
| Noisy neighbour | Per-tenant ingestion queue with a concurrency cap |
| Cost attribution | Tokens + embedding calls logged with `tenant_id` on every span |

## How you know it works

Run the full eval set on every pull request — a change that improves faithfulness while dropping context recall is a regression wearing a disguise.

| Metric | Meaning | Target |
|---|---|---|
| Context recall | Did retrieval fetch the gold chunk at all? | > 0.90 |
| Context precision | Are the retrieved chunks mostly relevant? | > 0.70 |
| Faithfulness | Is every claim supported by retrieved context? | > 0.95 |
| Answer correctness | Judge vs. hand-written answer | > 0.80 |
| Abstention rate | Says "not in the documents" on unanswerable questions | 5/5 |
| Citation validity | Cited chunk actually contains the claim | > 0.95 |
| Cost / query | Tokens + embedding + rerank | Track per question type |
| p95 latency | To first token | < 2s non-agentic, < 6s agentic |

### Failure modes you will actually hit

- **Ceiling you cannot see** — tuning prompts while context recall sits at 0.6 fixes nothing. Diagnose recall first.
- **The unanswerable trap** — models answer anyway from parametric knowledge, confidently. If you don't test for abstention, you don't have it.
- **Table amnesia** — most "the number is wrong" bugs trace to a table flattened during parsing.
- **Chunk size cargo cult** — no universally right size; sweep it against the golden set.
- **Reranker latency shock** — a cross-encoder over 40 candidates on CPU is fine; over 400 it isn't.
- **Stale index** — version documents, mark chunks with `doc_version`, make re-ingestion atomic.
- **Eval set contamination** — hold out 10 questions you never look at until release.

## Repo layout

```
docs/     source documents used to build the golden eval set
evals/    evals/golden.jsonl — hand-written questions + answers + page citations
main.py   FastAPI entrypoint
```

## Build plan & status

- [x] **0 — Golden set before code.** `evals/golden.jsonl`: 40 questions over 3 real PDFs in `docs/` — 20 single-hop, 10 multi-hop, 5 unanswerable, 5 table-lookup — each with a hand-written answer and source page.
- [x] **1 — Skeleton service.** FastAPI + uvicorn scaffolded (`main.py`, `pyproject.toml`).
- [ ] **2 — Deliberately naive baseline.** Text dump, fixed-size chunks, dense top-5, stuff-and-generate. Score it on the golden set and write the number here.
- [ ] **3 — Real parsing and chunking.** docling, typed blocks, heading paths, tables as atomic chunks, page + bbox. Citation viewer in Next.js.
- [ ] **4 — Hybrid retrieval + reranking.** BM25, RRF, cross-encoder over 40 candidates. Ablation table.
- [ ] **5 — Contextual chunks + entity graph.** Re-ingest with situating context; Kuzu graph retriever. Per-question-type score breakdown.
- [ ] **6 — Agentic loop + verification.** Assess/plan/execute with a hard hop budget; citation verifier. Accuracy-vs-cost curve.
- [ ] **7 — Service.** RLS multi-tenancy, per-tenant config/quotas, API keys, ingestion job status API, SSE streaming, tenant isolation test in CI.

## Getting started

```bash
uv sync
uv run fastapi dev main.py
```

## Reading list

- [infiniflow/ragflow](https://github.com/infiniflow/ragflow) — deep document understanding, GraphRAG, agentic workflow
- [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) — clearest small implementation of dual-level graph retrieval
- [circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) — PageRank-based graph exploration
- [SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) — agentic RAG as a production API
- [onyx-dot-app/onyx](https://github.com/onyx-dot-app/onyx) — enterprise search over connectors
- [explodinggradients/ragas](https://github.com/explodinggradients/ragas) — the evaluation library
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

## Where to take it

- **Visual retrieval** — ColPali-style page-image embeddings for layout-heavy documents (financial statements, engineering drawings, forms)
- **Live connectors** — Notion, Slack, Drive, GitHub with incremental sync and permission-mirrored retrieval
- **Feed the other projects** — Atlas as the shared retrieval backend for Sentinel's runbook lookup and Orchestra's design-pattern knowledge base
- **Publish the ablation** — same corpus, same golden set, Atlas vs. RAGFlow vs. LightRAG vs. plain top-k, with cost and latency columns
