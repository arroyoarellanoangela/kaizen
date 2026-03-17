<div align="center">

# Suyven RAG Engine

**The knowledge base that adapts to anything.**

*By [Suyven](https://suyven.com) · Built for production, not demos.*

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange?style=flat-square&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](LICENSE)

</div>

---

## What is this?

Most RAG engines are built to impress in demos. This one is built to run in production.

**Suyven RAG Engine** is the core infrastructure powering every Suyven product. It's GPU-first, domain-agnostic, and designed so that anyone — a startup, an enterprise, a solo dev — can plug in their knowledge and get a system that actually works. One engine. Unlimited domains. Zero compromise.

It's fast because it has to be. It's precise because anything less is useless. And it's modular because the future will require things we haven't thought of yet.

> *"The base layer of every intelligent system we'll ever build."* — Suyven

---

## Why it exists

Knowledge is scattered. Teams drown in documents, wikis, PDFs, codebases. LLMs hallucinate. Semantic search alone isn't enough.

Suyven RAG Engine exists because **retrieval is the hardest part**, and most solutions skip the hard work. We didn't.

- GPU-accelerated embeddings at 38x CPU speed
- Hybrid search combining semantic understanding with keyword precision
- Multi-domain isolation so your knowledge stays clean
- Auto-evaluation that flags failures before your users do

---

## Benchmarks

> Measured on NVIDIA RTX 5070 · CUDA 12.8 · PyTorch 2.12

| Metric | Result |
|--------|--------|
| NDCG@10 | **0.909** — 209-query ground-truth suite |
| Embedding throughput | **2,960 chunks/s** — 38x CPU baseline |
| FP16 VRAM savings | **−48%** vs FP32 (671 MB vs 1,290 MB) |
| FP16 retrieval fidelity | **99.3% Recall@10** — near-zero quality loss |
| Reranker latency | **8.8 ms/query** — FP16 on GPU |
| Test coverage | **209/209 passing** |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI 0.115 + SSE streaming + Uvicorn |
| Embeddings | BAAI/bge-m3 · 568M params · 1024-dim · multilingual · FP16 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 · FP16 on GPU |
| Vector Store | ChromaDB 1.5+ · cosine HNSW · multi-collection |
| Hybrid Search | BM25Okapi + Reciprocal Rank Fusion (RRF) |
| LLM | Any OpenAI-compatible endpoint · default: Groq llama-3.3-70b |
| GPU Monitoring | pynvml 13.0+ |
| Infrastructure | Docker · CPU + GPU variants |
| Runtime | Python 3.12 · PyTorch 2.12+ · CUDA 12.8 |

---

## Architecture

```
suyven-rag/
├── api.py                    # FastAPI + SSE streaming
├── app.py                    # Streamlit UI + GPU dashboard
├── ingest.py                 # CLI ingestion (3-phase pipeline)
├── query.py                  # CLI query tool
│
├── rag/                      # Core pipeline — 21 modules
│   ├── config.py             # Centralized config (env vars + Docker secrets)
│   ├── llm.py                # LLM abstraction (Ollama + OpenAI-compatible)
│   ├── agents.py             # 4-agent pipeline (Router → Retriever → Generator → Evaluator)
│   ├── orchestrator.py       # RoutePlan + hybrid search (BM25 + dense) + RRF fusion
│   ├── model_registry.py     # Embed/reranker singleton registry
│   ├── index_registry.py     # ChromaDB collection registry (static + dynamic)
│   ├── domain_registry.py    # Domain CRUD + isolation
│   ├── store.py              # Embedding + ChromaDB storage (FP16 GPU)
│   ├── pipeline.py           # Shared read+chunk pipeline
│   ├── chunker.py            # Character + paragraph + sentence chunking
│   ├── loader.py             # Multi-format reader (MD, TXT, PDF, PY, JSONL)
│   ├── eval.py               # Auto-evaluation + quality flagging + query log
│   ├── gap_tracker.py        # Knowledge gap analysis from query logs
│   ├── monitoring.py         # GPU telemetry via pynvml
│   ├── observability.py      # Structured logging + metrics + request tracing
│   ├── security.py           # Auth, CORS, rate limiting, input validation
│   ├── self_improve.py       # Auto-improvement from eval data
│   └── vector_store.py       # Vector store abstraction
│
├── finetune/                 # Embedding fine-tuning (LoRA, A/B testing)
├── tests/                    # 209-test pytest suite
├── benchmarks/               # Performance & quality benchmarks
├── docs/                     # Architecture, roadmap, benchmark reports
├── frontend/                 # React + Vite frontend
├── loadtest/                 # Locust load testing
├── scripts/                  # Deployment scripts
│
└── data/
    ├── chroma/               # ChromaDB persistent storage (HNSW)
    ├── domains/              # Domain configs + isolated indexes
    ├── eval/                 # Query evaluation logs (JSONL)
    └── knowledge/            # Source documents for ingestion
```

---

## How it works

### Ingestion — 3 phases

```
Documents → Parallel chunk (8 workers) → GPU embed (FP16) → ChromaDB index
```

Files are discovered automatically. Chunks are deduplicated by MD5. The embedding model loads once and stays warm.

### Query — 5 stages

```
Query → Route → Dense retrieval → BM25 → RRF fusion → Rerank → Answer
```

1. **Planning** — deterministic routing, no LLM calls, zero latency overhead
2. **Dense retrieval** — bge-m3 semantic search, 6x overfetch
3. **BM25** — keyword fallback, catches names, acronyms, exact matches
4. **Hybrid merge** — RRF fusion + source diversity cap (max 3 chunks/source)
5. **Cross-encoder reranking** — ms-marco-MiniLM, +7.0% Precision@5

Optional: LLM query expansion with late RRF fusion (~200ms on Groq).

### The 4-agent pipeline

| Agent | Role |
|-------|------|
| **Router** | Classifies complexity, picks strategy (dense / hybrid / category-filtered) |
| **Retriever** | Multi-tool reasoning — semantic search, entity search, sub-query decomposition, adjacent chunk expansion |
| **Generator** | Streams tokens, adapts prompt to retrieval quality in real time |
| **Evaluator** | Flags issues, escalates strategy on retry (dense → hybrid → no-category dense) |

---

## Key design decisions

| Decision | Why |
|----------|-----|
| FP16 embeddings | +109% throughput, −48% VRAM, 99.3% recall parity — no downside |
| Overfetch ×6 | Tested 4, 6, 8, 10 — 6 is optimal. 8+ hurts NDCG |
| Max 3 chunks/source | Prevents one document from dominating context. 2 was too aggressive |
| ms-marco reranker | Outperforms bge-reranker-v2 on this corpus |
| RRF over score averaging | Rank-based, robust against score scale differences across retrievers |
| Groq for LLM | 70B quality at speed. GPU VRAM stays reserved for embed + reranker |
| Domain isolation | Specialized indexes prevent cross-domain contamination |
| ReACT retriever | Heuristic reasoning — catches entity queries without LLM cost |
| No fine-tuning yet | Evidence-gated. Only when NDCG plateaus on real production queries |

---

## Quick start

```bash
# Clone
git clone https://github.com/suyven-core/rag-engine
cd rag-engine

# Configure
cp .env.example .env
# Edit .env with your LLM API key

# Run (GPU)
docker compose -f docker-compose.gpu.yml up

# Ingest your knowledge
python ingest.py --domain my-domain --path ./data/knowledge/

# Query
python query.py --domain my-domain "What is...?"
```

---

## Configuration

```env
# Embeddings
EMBED_MODEL=BAAI/bge-m3
EMBED_BATCH=256

# Chunking
CHUNK_SIZE=600
CHUNK_OVERLAP=80

# Retrieval
TOP_K=5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
OVERFETCH_FACTOR=6
RERANKER_BATCH_SIZE=32

# LLM
LLM_PROVIDER=openai
LLM_MODEL=llama-3.3-70b-versatile
LLM_API_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<your-key>

# Fallback LLM
FALLBACK_PROVIDER=openai
FALLBACK_MODEL=gemini-2.5-flash
FALLBACK_API_URL=https://generativelanguage.googleapis.com/v1beta/openai
FALLBACK_API_KEY=<your-key>

# Security
AUTH_ENABLED=false
API_KEYS=key1,key2
RATE_LIMIT_RPM=60
RATE_LIMIT_BURST=10
MAX_QUERY_LENGTH=2000
MAX_TOP_K=20
CORS_ORIGINS=http://localhost:5173

# Observability
LOG_FORMAT=text   # use "json" in production
WORKERS=8
```

Docker secrets at `/run/secrets/<NAME>` override all env vars.

---

## API reference

```python
# Orchestrator
plan(query, category, top_k) → RoutePlan
execute_search(query, route, category, use_expansion) → [results]
format_context(results) → str

# Embeddings & storage
get_embed_model() → SentenceTransformer
embed_batch(texts) → [[float], ...]
add_chunks(col, path, chunks, knowledge_dir) → (added, skipped)

# Registries
get_index(name) → chromadb.Collection
get_embed_model(name) → SentenceTransformer
get_reranker(name) → CrossEncoder

# Domains
create_domain(name, description, system_prompt, categories) → DomainConfig
get_domain(slug) → DomainConfig
list_domains() → [DomainConfig]

# Evaluation
compute_flags(record) → [flags]
log_eval(record) → None
analyze_gaps(entries, top_n) → GapReport

# LLM
quick_complete(prompt, ...) → str
stream_chat(query, context, system_prompt, ...) → Generator[str]
```

---

## Roadmap

- [ ] Multi-modal ingestion (images, tables, charts)
- [ ] Embedding fine-tuning when production data justifies it
- [ ] GraphRAG layer for entity-relationship queries
- [ ] REST API authentication + multi-tenant key management
- [ ] Hosted version — plug in your docs, get an endpoint

---

<div align="center">

**Built by [Suyven](https://suyven.com)**

*The base layer. Everything else is built on top of this.*

[angela@suyven.com](mailto:angela@suyven.com)

</div>
