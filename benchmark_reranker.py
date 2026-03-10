#!/usr/bin/env python3
"""
Cross-Encoder Reranker Benchmark
==================================
Measures reranker performance across precision/batch configurations
and end-to-end retrieval latency (bi-encoder + reranker).

Metrics:
- Reranker latency per query (ms)
- Pairs/second throughput
- End-to-end query latency (retrieve + rerank)
- Peak VRAM
- FP16 vs FP32 comparison

Usage:
    python benchmark_reranker.py              # full benchmark
    python benchmark_reranker.py --quick      # fewer queries, smaller candidate set
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Ensure UTF-8 output on Windows (cp1252 can't handle box-drawing/emoji chars)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from sentence_transformers import CrossEncoder

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, KNOWLEDGE_DIR
from rag.chunker import chunk_text
from rag.loader import iter_files, read_file
from rag.store import get_collection, get_embed_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NUM_RUNS = 3
WARMUP_QUERIES = 3

# Queries that exercise different domains in the knowledge base
BENCH_QUERIES = [
    "What is a star schema and how does it differ from snowflake?",
    "How does medallion architecture work in data lakehouse?",
    "Explain RAG orchestration patterns with LangGraph",
    "What are GPU optimization techniques for transformer inference?",
    "How to design a dimensional model for analytics?",
    "Explain chunking strategies for embedding pipelines",
    "What is schema evolution and how to handle it?",
    "How does vector database indexing work with HNSW?",
    "What are best practices for data pipeline testing?",
    "Explain the difference between batch and stream processing",
    "How to implement data quality checks in production?",
    "What is prompt engineering for RAG systems?",
    "Explain Redshift table design and distribution keys",
    "How does Apache Spark handle partitioning?",
    "What are sampling methods in statistics?",
]


@dataclass
class RerankerConfig:
    name: str
    device: str       # "cpu" or "cuda"
    precision: str    # "fp32" or "fp16"
    batch_size: int
    n_candidates: int  # how many candidates to rerank per query


@dataclass
class RerankerResult:
    config: str
    device: str
    precision: str
    batch_size: int
    n_candidates: int
    n_queries: int
    mean_latency_ms: float      # avg ms per query (rerank only)
    std_latency_ms: float
    mean_pairs_per_sec: float   # pairs/s throughput
    std_pairs_per_sec: float
    mean_e2e_ms: float          # end-to-end: retrieve + rerank
    std_e2e_ms: float
    peak_vram_mb: float
    speedup: float = 1.0
    runs: list[dict] = None


# Base configs
BASE_CONFIGS = [
    RerankerConfig("CPU FP32",  "cpu",  "fp32", 32,  20),
    RerankerConfig("GPU FP32",  "cuda", "fp32", 32,  20),
    RerankerConfig("GPU FP16",  "cuda", "fp16", 32,  20),
]

# FP16 batch sweep
FP16_BATCH_SWEEP = [8, 16, 32, 64, 128]

# Candidate count sweep (how many docs to rerank)
CANDIDATE_SWEEP = [10, 20, 40, 60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def load_reranker(device: str, precision: str) -> CrossEncoder:
    """Load cross-encoder with specified device/precision."""
    model = CrossEncoder(RERANKER_MODEL)
    if device == "cuda" and torch.cuda.is_available():
        model.model = model.model.to("cuda")
        if precision == "fp16":
            model.model = model.model.half()
    return model


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_reranker_benchmark(
    reranker: CrossEncoder,
    queries: list[str],
    corpus_chunks: list[str],
    cfg: RerankerConfig,
    collection,
) -> dict:
    """
    Run one full pass: for each query, retrieve candidates then rerank.
    Returns per-query timing breakdown.
    """
    query_timings = []

    for query in queries:
        # Stage 1 — bi-encoder retrieval (via ChromaDB)
        t_retrieve_start = time.perf_counter()
        results = collection.query(
            query_texts=[query],
            n_results=min(cfg.n_candidates, collection.count()),
        )
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t_retrieve_end = time.perf_counter()

        if not results or not results["documents"] or not results["documents"][0]:
            continue

        candidates = results["documents"][0]

        # Stage 2 — cross-encoder reranking
        pairs = [[query, doc] for doc in candidates]

        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t_rerank_start = time.perf_counter()
        reranker.predict(pairs, batch_size=cfg.batch_size, show_progress_bar=False)
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t_rerank_end = time.perf_counter()

        retrieve_ms = (t_retrieve_end - t_retrieve_start) * 1000
        rerank_ms = (t_rerank_end - t_rerank_start) * 1000
        n_pairs = len(pairs)

        query_timings.append({
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "e2e_ms": retrieve_ms + rerank_ms,
            "n_pairs": n_pairs,
            "pairs_per_sec": n_pairs / (rerank_ms / 1000) if rerank_ms > 0 else 0,
        })

    return query_timings


# ---------------------------------------------------------------------------
# Full benchmark for one config
# ---------------------------------------------------------------------------

def benchmark_config(
    cfg: RerankerConfig,
    queries: list[str],
    corpus_chunks: list[str],
    collection,
) -> RerankerResult | None:
    """Run NUM_RUNS passes for one config, aggregate stats."""
    print(f"\n{'-'*60}")
    print(f"  {cfg.name}  |  batch={cfg.batch_size}  |  candidates={cfg.n_candidates}")
    print(f"{'-'*60}")

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA not available — skipping")
        return None

    clear_gpu()

    # Load reranker
    print("  Loading reranker...")
    reranker = load_reranker(cfg.device, cfg.precision)

    # Warmup
    print(f"  Warmup ({WARMUP_QUERIES} queries)...")
    run_reranker_benchmark(reranker, queries[:WARMUP_QUERIES], corpus_chunks, cfg, collection)

    if cfg.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed runs
    all_runs = []
    for i in range(NUM_RUNS):
        timings = run_reranker_benchmark(reranker, queries, corpus_chunks, cfg, collection)

        avg_rerank = sum(t["rerank_ms"] for t in timings) / len(timings)
        avg_pps = sum(t["pairs_per_sec"] for t in timings) / len(timings)
        avg_e2e = sum(t["e2e_ms"] for t in timings) / len(timings)

        all_runs.append({
            "avg_rerank_ms": avg_rerank,
            "avg_pairs_per_sec": avg_pps,
            "avg_e2e_ms": avg_e2e,
            "timings": timings,
        })
        print(f"  Run {i+1}/{NUM_RUNS}: {avg_rerank:.1f} ms/query | {avg_pps:.0f} pairs/s | e2e: {avg_e2e:.1f} ms")

    peak_vram = get_peak_vram_mb()

    # Aggregate across runs
    latencies = [r["avg_rerank_ms"] for r in all_runs]
    pps_list = [r["avg_pairs_per_sec"] for r in all_runs]
    e2e_list = [r["avg_e2e_ms"] for r in all_runs]

    def mean(xs): return sum(xs) / len(xs)
    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    result = RerankerResult(
        config=cfg.name,
        device=cfg.device,
        precision=cfg.precision,
        batch_size=cfg.batch_size,
        n_candidates=cfg.n_candidates,
        n_queries=len(queries),
        mean_latency_ms=round(mean(latencies), 2),
        std_latency_ms=round(std(latencies), 2),
        mean_pairs_per_sec=round(mean(pps_list), 0),
        std_pairs_per_sec=round(std(pps_list), 0),
        mean_e2e_ms=round(mean(e2e_list), 2),
        std_e2e_ms=round(std(e2e_list), 2),
        peak_vram_mb=round(peak_vram, 0),
        runs=[{"latency_ms": r["avg_rerank_ms"], "pairs_per_sec": r["avg_pairs_per_sec"]} for r in all_runs],
    )

    print(f"  → Rerank: {result.mean_latency_ms:.1f} ± {result.std_latency_ms:.1f} ms/query")
    print(f"  → Throughput: {result.mean_pairs_per_sec:.0f} ± {result.std_pairs_per_sec:.0f} pairs/s")
    print(f"  → E2E: {result.mean_e2e_ms:.1f} ± {result.std_e2e_ms:.1f} ms/query")
    print(f"  → Peak VRAM: {result.peak_vram_mb:.0f} MB")

    del reranker
    clear_gpu()

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    base_results: list[RerankerResult],
    batch_results: list[RerankerResult],
    candidate_results: list[RerankerResult],
) -> str:
    """Generate markdown report."""
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    # Calculate speedups relative to CPU
    cpu_lat = next((r.mean_latency_ms for r in base_results if r.device == "cpu"), None)
    all_results = base_results + batch_results + candidate_results
    if cpu_lat and cpu_lat > 0:
        for r in all_results:
            r.speedup = round(cpu_lat / r.mean_latency_ms, 1) if r.mean_latency_ms > 0 else 1.0

    report = f"""# Reranker Benchmark Results — Cross-Encoder Performance

## System Info

| Component | Value |
|-----------|-------|
| **GPU** | {gpu_name} |
| **Reranker Model** | `{RERANKER_MODEL}` |
| **Embedding Model** | `{EMBED_MODEL}` |
| **Queries** | {base_results[0].n_queries if base_results else 'N/A'} |
| **Runs per Config** | {NUM_RUNS} |
| **PyTorch** | {torch.__version__} |
| **CUDA** | {torch.version.cuda if torch.cuda.is_available() else 'N/A'} |

## Core Results — CPU vs GPU vs Precision

Reranking {base_results[0].n_candidates if base_results else 20} candidates per query.

| Config | Rerank Latency | Throughput | E2E Latency | VRAM | Speedup |
|--------|---------------|------------|-------------|------|---------|
"""
    for r in base_results:
        report += (
            f"| {r.config} | **{r.mean_latency_ms:.1f}** ± {r.std_latency_ms:.1f} ms | "
            f"{r.mean_pairs_per_sec:.0f} pairs/s | "
            f"{r.mean_e2e_ms:.1f} ms | {r.peak_vram_mb:.0f} MB | **{r.speedup}x** |\n"
        )

    if batch_results:
        report += f"""
## FP16 Batch Size Sweep

| Batch | Rerank Latency | Throughput | VRAM | Speedup |
|-------|---------------|------------|------|---------|
"""
        for r in batch_results:
            report += (
                f"| {r.batch_size} | **{r.mean_latency_ms:.1f}** ± {r.std_latency_ms:.1f} ms | "
                f"{r.mean_pairs_per_sec:.0f} pairs/s | "
                f"{r.peak_vram_mb:.0f} MB | **{r.speedup}x** |\n"
            )

    if candidate_results:
        report += f"""
## Candidate Count Scaling

How reranker latency scales with the number of candidates to rerank (GPU FP16).

| Candidates | Rerank Latency | Throughput | E2E Latency |
|-----------|---------------|------------|-------------|
"""
        for r in candidate_results:
            report += (
                f"| {r.n_candidates} | **{r.mean_latency_ms:.1f}** ms | "
                f"{r.mean_pairs_per_sec:.0f} pairs/s | "
                f"{r.mean_e2e_ms:.1f} ms |\n"
            )

    # Key observations
    report += "\n## Key Observations\n\n"

    cpu_r = next((r for r in base_results if r.device == "cpu"), None)
    fp32_r = next((r for r in base_results if r.precision == "fp32" and r.device == "cuda"), None)
    fp16_r = next((r for r in base_results if r.precision == "fp16"), None)

    if cpu_r and fp32_r:
        report += f"1. **CPU → GPU FP32**: {fp32_r.speedup}x faster ({cpu_r.mean_latency_ms:.1f} → {fp32_r.mean_latency_ms:.1f} ms/query)\n"
    if fp32_r and fp16_r:
        lat_pct = ((fp32_r.mean_latency_ms - fp16_r.mean_latency_ms) / fp32_r.mean_latency_ms * 100)
        report += f"2. **FP32 → FP16**: {lat_pct:.0f}% latency reduction ({fp32_r.mean_latency_ms:.1f} → {fp16_r.mean_latency_ms:.1f} ms)\n"
    if candidate_results:
        report += f"3. **Scaling**: Latency from {candidate_results[0].mean_latency_ms:.1f} ms ({candidate_results[0].n_candidates} candidates) to {candidate_results[-1].mean_latency_ms:.1f} ms ({candidate_results[-1].n_candidates} candidates)\n"

    report += f"""
## Methodology

- **Reranker**: Cross-encoder scores each (query, document) pair independently
- **End-to-end**: ChromaDB bi-encoder retrieval + cross-encoder reranking
- **Warmup**: {WARMUP_QUERIES} queries discarded before timing
- **Repetitions**: {NUM_RUNS} full passes, mean ± std reported
- **Timing**: `torch.cuda.synchronize()` for accurate GPU measurement
- **Isolation**: GPU memory cleared between configurations

---
*Generated by `benchmark_reranker.py` — Kaizen RAG GPU Optimization Pipeline*
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kaizen RAG Reranker Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer queries)")
    args = parser.parse_args()

    queries = BENCH_QUERIES[:5] if args.quick else BENCH_QUERIES

    print("=" * 60)
    print("  KAIZEN RAG — RERANKER BENCHMARK")
    print("=" * 60)
    print(f"  Reranker:   {RERANKER_MODEL}")
    print(f"  Queries:    {len(queries)}")
    print(f"  Runs/cfg:   {NUM_RUNS}")
    print(f"  CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Get collection (needs data indexed)
    col = get_collection()
    chunk_count = col.count()
    if chunk_count == 0:
        print("\n  ❌ Knowledge base empty. Run: python ingest.py")
        return
    print(f"\n  Knowledge base: {chunk_count:,} chunks")

    # Load dummy corpus for reference
    corpus_chunks = []  # not directly needed, ChromaDB handles retrieval

    # -- Part 1: Base configs ---------------------------------------
    print(f"\n\n{'='*60}")
    print("  PART 1: CORE BENCHMARKS (20 candidates)")
    print(f"{'='*60}")

    base_results = []
    for cfg in BASE_CONFIGS:
        result = benchmark_config(cfg, queries, corpus_chunks, col)
        if result:
            base_results.append(result)

    # -- Part 2: FP16 batch size sweep ------------------------------
    print(f"\n\n{'='*60}")
    print("  PART 2: FP16 BATCH SIZE SWEEP")
    print(f"{'='*60}")

    batch_results = []
    for bs in FP16_BATCH_SWEEP:
        cfg = RerankerConfig(f"FP16 batch={bs}", "cuda", "fp16", bs, 20)
        result = benchmark_config(cfg, queries, corpus_chunks, col)
        if result:
            batch_results.append(result)

    # -- Part 3: Candidate count scaling ----------------------------
    print(f"\n\n{'='*60}")
    print("  PART 3: CANDIDATE COUNT SCALING (GPU FP16)")
    print(f"{'='*60}")

    candidate_results = []
    for nc in CANDIDATE_SWEEP:
        cfg = RerankerConfig(f"FP16 n={nc}", "cuda", "fp16", 32, nc)
        result = benchmark_config(cfg, queries, corpus_chunks, col)
        if result:
            candidate_results.append(result)

    # -- Generate report --------------------------------------------
    print(f"\n\n{'='*60}")
    print("  GENERATING REPORT")
    print(f"{'='*60}")

    report = generate_report(base_results, batch_results, candidate_results)
    output_path = Path("RERANKER_BENCHMARK.md")
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  📄 Report: {output_path.absolute()}")

    # Raw JSON
    json_path = Path("benchmark_reranker_raw.json")
    all_results = base_results + batch_results + candidate_results
    json_path.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"  📊 Raw data: {json_path.absolute()}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<25} {'Latency':>10} {'Pairs/s':>10} {'E2E':>10} {'VRAM':>8} {'Speed':>6}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
    for r in all_results:
        print(
            f"  {r.config:<25} {r.mean_latency_ms:>8.1f}ms {r.mean_pairs_per_sec:>8.0f} "
            f"{r.mean_e2e_ms:>8.1f}ms {r.peak_vram_mb:>6.0f}MB {r.speedup:>5.1f}x"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
