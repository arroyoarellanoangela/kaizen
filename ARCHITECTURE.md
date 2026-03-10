# Architecture — Kaizen RAG Pipeline

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Documents (MD/TXT/PDF)                                        │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐    ThreadPoolExecutor (8 workers)               │
│   │  Loader   │──── Parallel file reading                       │
│   └───────────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐    600-char chunks, 80-char overlap             │
│   │  Chunker  │──── Character-based text splitting              │
│   └───────────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────────────────────────────┐                      │
│   │     GPU Embedding (FP16)             │                      │
│   │                                      │                      │
│   │  Model: all-MiniLM-L6-v2            │                      │
│   │  Precision: FP16 (half)             │                      │
│   │  Batch size: 256 (benchmark-optimal)│                      │
│   │  Device: CUDA (RTX 5070)            │                      │
│   │                                      │                      │
│   │  Throughput: ~2960 chunks/s          │                      │
│   │  Speedup: 38× vs CPU               │                      │
│   │  VRAM: 671 MB (−48% vs FP32)        │                      │
│   └──────────────────────────────────────┘                      │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────┐    Batch inserts (500 chunks/call)              │
│   │ ChromaDB  │──── Persistent vector store                     │
│   └───────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query                                                    │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────┐    Same FP16 model (cached in GPU)           │
│   │ Query Embed  │──── Single embedding, <1ms                   │
│   └──────────────┘                                              │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────┐    Cosine similarity — overfetch ×4          │
│   │  Stage 1:    │──── Bi-encoder fast retrieval                │
│   │  Retriever   │    Top-4k candidates from ChromaDB           │
│   └──────────────┘                                              │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────────────────────────────┐                      │
│   │     Stage 2: Cross-Encoder (FP16)   │                      │
│   │                                      │                      │
│   │  Model: ms-marco-MiniLM-L-6-v2     │                      │
│   │  Precision: FP16 (half)             │                      │
│   │  Device: CUDA (RTX 5070)            │                      │
│   │                                      │                      │
│   │  Pairwise (query, chunk) scoring    │                      │
│   │  Re-ranks candidates → top-k        │                      │
│   └──────────────────────────────────────┘                      │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────┐    Context = reranked top-k chunks           │
│   │  LLM (Ollama)│──── qwen3:8b streaming response              │
│   └──────────────┘                                              │
│         │                                                       │
│         ▼                                                       │
│   Synthesized Answer + Source References                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     GPU METRICS DASHBOARD                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   pynvml (NVML bindings)                                        │
│         │                                                       │
│         ▼                                                       │
│   ┌──────────────────────────────────────┐                      │
│   │  Real-time GPU Monitoring            │                      │
│   │                                      │                      │
│   │  • Device name (RTX 5070)           │                      │
│   │  • Temperature (°C)                 │                      │
│   │  • GPU utilization (%)              │                      │
│   │  • VRAM used / total (GB + bar)     │                      │
│   └──────────────────────────────────────┘                      │
│         │                                                       │
│         ▼                                                       │
│   Streamlit sidebar — updates each page interaction             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale | Validated By |
|----------|-----------|--------------|
| FP16 bi-encoder | +109% throughput, −48% VRAM | `BENCHMARK_RESULTS.md` |
| Batch size = 256 | Optimal from sweep (256→2048) | `benchmark.py` |
| FP16 quality validated | 99.3% Recall@10 maintained | `QUALITY_VALIDATION.md` |
| 2-stage retrieval | Cross-encoder reranks bi-encoder candidates | `RERANKER_QUALITY.md` |
| Overfetch ×4 | Ensures reranker has enough candidates to promote | Retrieval experiments |
| FP16 cross-encoder | GPU-accelerated pairwise reranking | `RERANKER_BENCHMARK.md` |
| Parallel file I/O | Eliminates read bottleneck | Phase 1 timing |
| Ollama for LLM only | GPU VRAM reserved for embeddings + reranker | Architecture split |
| sentence-transformers | Direct GPU control, no HTTP overhead | vs Ollama API |
| pynvml GPU dashboard | Real-time VRAM / temp / utilization monitoring | Sidebar UI |

## File Structure

```
kaizen-v1/
├── app.py                      # Streamlit UI (ingest + query + LLM + GPU dashboard)
├── ingest.py                   # CLI ingestion
├── benchmark.py                # Bi-encoder GPU performance benchmark
├── benchmark_reranker.py       # Cross-encoder reranker benchmark
├── validate_quality.py         # FP32 vs FP16 embedding fidelity
├── validate_reranker.py        # Reranker quality (NDCG, ground truth)
├── BENCHMARK_RESULTS.md        # Bi-encoder performance report
├── RERANKER_BENCHMARK.md       # Reranker performance report (generated)
├── QUALITY_VALIDATION.md       # FP16 fidelity report
├── RERANKER_QUALITY.md         # Reranker quality report (generated)
├── ARCHITECTURE.md             # This file
├── rag/
│   ├── config.py               # Configuration (env vars)
│   ├── loader.py               # File reader (MD, TXT, PDF)
│   ├── chunker.py              # Text chunking
│   ├── store.py                # Embedding + ChromaDB (FP16 GPU)
│   └── retriever.py            # 2-stage search (bi-encoder + cross-encoder)
└── data/
    └── chroma/                 # Persistent vector DB
```
