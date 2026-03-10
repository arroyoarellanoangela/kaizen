# Kaizen v1 - GPU-Optimized RAG Pipeline

> **Scope**: Kaizen v1 is the **retrieval and ranking engine** — ingestion, embedding, search, reranking, and LLM response generation. The frontend application lives in a separate project and will consume this engine via API.

A retrieval-augmented generation system built to demonstrate GPU optimization techniques on NVIDIA hardware. Every design decision is benchmarked, every optimization is validated, and every claim has reproducible evidence.

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Embedding throughput | **2,960 chunks/s** (38x vs CPU) | [Benchmark](BENCHMARK_RESULTS.md) |
| FP16 VRAM savings | **-48%** (671 MB vs 1,290 MB) | [Benchmark](BENCHMARK_RESULTS.md) |
| FP16 retrieval fidelity | **99.3% Recall@10** | [Validation](QUALITY_VALIDATION.md) |
| Reranker latency (FP16) | **8.8 ms/query** (1.7x vs CPU) | [Reranker Benchmark](RERANKER_BENCHMARK.md) |
| Reranker quality gain | **+7.0% Precision@5** | [Reranker Quality](RERANKER_QUALITY.md) |

**Hardware**: NVIDIA GeForce RTX 5070 (12 GB VRAM) | PyTorch 2.12 | CUDA 12.8

## What This Project Demonstrates

### 1. GPU-Optimized Embedding Pipeline

The bi-encoder embedding layer (`all-MiniLM-L6-v2`) runs on CUDA with FP16 mixed precision, achieving 38x throughput over CPU. Batch size was determined through a sweep of 256-2048, with 256 optimal for this model and GPU.

FP16 was not assumed safe - it was validated. A dedicated script compares FP32 and FP16 retrieval results across 15 queries on a 5,000-chunk corpus, measuring both result overlap (Recall@10 = 99.3%) and ranking agreement (Kendall Tau = 0.99).

### 2. Two-Stage Retrieval with Measured Quality Gains

The query pipeline uses a two-stage approach:
- **Stage 1**: Bi-encoder cosine similarity via ChromaDB (fast, overfetch x4)
- **Stage 2**: Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`, FP16 on CUDA)

The reranker was validated against 23 manually curated ground-truth queries, including 5 intentionally tricky cases (ambiguous terms, indirect phrasing, cross-domain, similar-document discrimination). Results: NDCG@5 improved from 0.816 to 0.865, with 6 queries improved, 15 unchanged, and 2 regressed.

The regressions are documented, not hidden.

### 3. Benchmark-First Methodology

Every performance claim is backed by:
- **Warmup phases** to exclude CUDA initialization from measurements
- **Multiple runs** (3 per config) with mean and standard deviation
- **`torch.cuda.synchronize()`** for accurate GPU timing
- **`torch.cuda.max_memory_allocated()`** for peak VRAM tracking
- **GPU memory isolation** between configurations (`empty_cache` + `reset_peak_memory_stats`)
- **Raw JSON** alongside markdown reports for reproducibility

### 4. Real-Time GPU Observability

The Streamlit interface includes a live GPU dashboard (via `pynvml`/NVML) showing device name, temperature, utilization percentage, and VRAM usage with a progress bar. This provides visibility into GPU resource consumption during ingestion and query workloads.

## Architecture

```
Documents --> Parallel I/O --> Chunking --> GPU Embedding (FP16)
                                              |
                                              v
                                           ChromaDB
                                              |
           User Query --> Bi-Encoder --> Overfetch x4
                                              |
                                              v
                                   Cross-Encoder Rerank (FP16)
                                              |
                                              v
                                     LLM (Ollama qwen3:8b)
                                              |
                                              v
                                      Answer + Sources
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram with component details.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Index your knowledge base
python ingest.py --dir /path/to/documents

# Launch web UI
streamlit run app.py

# Or query from CLI
python query.py "What is a star schema?"
```

## Reproduce the Benchmarks

```bash
# Embedding pipeline: CPU vs GPU vs FP16, batch sweep
python benchmark.py

# Reranker: CPU vs GPU, FP16 vs FP32, batch sweep, candidate scaling
python benchmark_reranker.py

# FP16 embedding quality: Recall@10, Kendall Tau
python validate_quality.py

# Reranker quality: NDCG@5/10, Precision@5, ground truth
python validate_reranker.py --side-by-side
```

## Project Structure

```
kaizen-v1/
|-- app.py                      # Streamlit UI + GPU dashboard
|-- ingest.py                   # CLI ingestion (parallel I/O)
|-- benchmark.py                # Bi-encoder GPU benchmark
|-- benchmark_reranker.py       # Cross-encoder benchmark
|-- validate_quality.py         # FP16 fidelity validation
|-- validate_reranker.py        # Reranker quality validation
|-- rag/
|   |-- config.py               # Settings (.env)
|   |-- loader.py               # File reader (MD, TXT, PDF)
|   |-- chunker.py              # Text splitting + overlap
|   |-- store.py                # Embedding + ChromaDB (FP16 CUDA)
|   |-- retriever.py            # 2-stage retrieval (bi + cross-encoder)
|-- BENCHMARK_RESULTS.md        # Embedding performance report
|-- RERANKER_BENCHMARK.md       # Reranker performance report
|-- QUALITY_VALIDATION.md       # FP16 fidelity report
|-- RERANKER_QUALITY.md         # Reranker quality report
|-- ARCHITECTURE.md             # System design + decisions
|-- KNOWN_LIMITATIONS.md        # Documented issues + edge cases
|-- ROADMAP.md                  # Future work
```

## Reports

| Report | What It Measures |
|--------|-----------------|
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | CPU vs GPU FP32 vs GPU FP16 embedding throughput and VRAM |
| [QUALITY_VALIDATION.md](QUALITY_VALIDATION.md) | FP32 vs FP16 retrieval fidelity (Recall@10, Kendall Tau) |
| [RERANKER_BENCHMARK.md](RERANKER_BENCHMARK.md) | Cross-encoder latency, throughput, batch/candidate scaling |
| [RERANKER_QUALITY.md](RERANKER_QUALITY.md) | Reranker NDCG@5/10 improvement over bi-encoder baseline |

## Tech Stack

- **Embedding**: `sentence-transformers` (direct CUDA, no HTTP overhead)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (FP16 on GPU)
- **Vector Store**: ChromaDB (persistent, cosine similarity, HNSW)
- **LLM**: Ollama (`qwen3:8b`, streaming)
- **GPU Monitoring**: `pynvml` (NVML bindings)
- **UI**: Streamlit
- **Hardware**: NVIDIA RTX 5070, CUDA 12.8, PyTorch 2.12

## Next Steps

1. **Connect frontend** to the Kaizen RAG backend (API layer between UI and retrieval engine)
2. **User testing** with real queries to surface issues not caught by automated evaluation
3. **Improve PDF chunking** to preserve table structure
4. **Evaluate multi-index** if single-index quality degrades as corpus grows

See [ROADMAP.md](ROADMAP.md) for full plan and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for documented issues.
