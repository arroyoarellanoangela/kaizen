#!/usr/bin/env python3
"""
GPU-Optimized Embedding Benchmark
==================================
Benchmark matrix comparing CPU vs GPU, FP32 vs FP16, and batch size tuning
for transformer-based embedding inference.

Features:
- 4 base configs + FP16 batch size sweep
- 3 repetitions per config with mean/stddev
- Proper CUDA warmup
- Peak VRAM measurement
- Auto-generated markdown report

Usage:
    python benchmark.py                 # full benchmark (10K chunks)
    python benchmark.py --quick         # quick test (1K chunks)
    python benchmark.py --chunks 5000   # custom chunk count
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from rag.chunker import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, KNOWLEDGE_DIR
from rag.loader import iter_files, read_file


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_RUNS = 3         # repetitions per config
WARMUP_CHUNKS = 256  # discarded before timing

@dataclass
class BenchmarkConfig:
    name: str
    device: str          # "cpu" or "cuda"
    precision: str       # "fp32" or "fp16"
    batch_size: int


@dataclass
class RunResult:
    total_sec: float
    chunks_per_sec: float
    peak_vram_mb: float


@dataclass
class BenchmarkResult:
    config: str
    device: str
    precision: str
    batch_size: int
    num_chunks: int
    mean_throughput: float
    std_throughput: float
    mean_sec: float
    std_sec: float
    sec_per_10k: float
    peak_vram_mb: float
    model_vram_mb: float
    speedup: float = 1.0
    runs: list[float] = None  # individual throughputs


# Base configs
BASE_CONFIGS = [
    BenchmarkConfig("CPU FP32",      "cpu",  "fp32", 32),
    BenchmarkConfig("GPU FP32",      "cuda", "fp32", 256),
    BenchmarkConfig("GPU FP16",      "cuda", "fp16", 256),
]

# FP16 batch sweep — find the sweet spot
FP16_BATCH_SWEEP = [256, 512, 1024, 2048]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample_chunks(max_chunks: int) -> list[str]:
    """Load real chunks from the knowledge base."""
    print(f"  Loading chunks from {KNOWLEDGE_DIR}...")
    all_chunks = []
    for path in iter_files(KNOWLEDGE_DIR):
        text = read_file(path)
        if not text.strip():
            continue
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        if len(all_chunks) >= max_chunks:
            break
    all_chunks = all_chunks[:max_chunks]
    print(f"  Loaded {len(all_chunks)} chunks")
    return all_chunks


def clear_gpu():
    """Clear GPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    model: SentenceTransformer,
    chunks: list[str],
    batch_size: int,
    device: str,
) -> RunResult:
    """Execute one timed encoding run."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t_start = time.perf_counter()
    model.encode(chunks, batch_size=batch_size, show_progress_bar=False, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    total_sec = t_end - t_start
    return RunResult(
        total_sec=total_sec,
        chunks_per_sec=len(chunks) / total_sec,
        peak_vram_mb=get_peak_vram_mb(),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    cfg: BenchmarkConfig,
    chunks: list[str],
    num_runs: int = NUM_RUNS,
) -> BenchmarkResult | None:
    """Run a full benchmark for one configuration (warmup + N runs)."""
    print(f"\n{'─'*60}")
    print(f"  {cfg.name}  |  {cfg.device}  |  {cfg.precision}  |  batch={cfg.batch_size}")
    print(f"{'─'*60}")

    # Skip GPU configs if no CUDA
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA not available — skipping")
        return None

    clear_gpu()

    # Load model
    print(f"  Loading model...")
    model = SentenceTransformer(EMBED_MODEL)
    if cfg.device == "cuda":
        model = model.to("cuda")
    if cfg.precision == "fp16" and cfg.device == "cuda":
        model = model.half()

    model_vram = get_vram_mb()

    # Warmup (discarded)
    print(f"  Warmup ({WARMUP_CHUNKS} chunks)...")
    warmup = chunks[:WARMUP_CHUNKS]
    model.encode(warmup, batch_size=cfg.batch_size, show_progress_bar=False, device=cfg.device)
    if cfg.device == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    run_results = []
    for i in range(num_runs):
        r = run_single(model, chunks, cfg.batch_size, cfg.device)
        run_results.append(r)
        print(f"  Run {i+1}/{num_runs}: {r.chunks_per_sec:.0f} chunks/s ({r.total_sec:.2f}s)")

    # Aggregate
    throughputs = [r.chunks_per_sec for r in run_results]
    times = [r.total_sec for r in run_results]
    mean_tp = sum(throughputs) / len(throughputs)
    std_tp = (sum((t - mean_tp)**2 for t in throughputs) / len(throughputs)) ** 0.5
    mean_sec = sum(times) / len(times)
    std_sec = (sum((t - mean_sec)**2 for t in times) / len(times)) ** 0.5
    peak_vram = max(r.peak_vram_mb for r in run_results)

    print(f"  → Mean: {mean_tp:.0f} ± {std_tp:.0f} chunks/s  |  VRAM: {peak_vram:.0f} MB")

    # Cleanup
    del model
    clear_gpu()

    return BenchmarkResult(
        config=cfg.name,
        device=cfg.device,
        precision=cfg.precision,
        batch_size=cfg.batch_size,
        num_chunks=len(chunks),
        mean_throughput=round(mean_tp, 1),
        std_throughput=round(std_tp, 1),
        mean_sec=round(mean_sec, 3),
        std_sec=round(std_sec, 3),
        sec_per_10k=round(10_000 / mean_tp, 1) if mean_tp > 0 else 0,
        peak_vram_mb=round(peak_vram, 0),
        model_vram_mb=round(model_vram, 0),
        runs=[round(t, 1) for t in throughputs],
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    base_results: list[BenchmarkResult],
    sweep_results: list[BenchmarkResult],
    num_chunks: int,
) -> str:
    """Generate markdown benchmark report."""
    # Calculate speedups relative to CPU baseline
    cpu_tp = next((r.mean_throughput for r in base_results if r.device == "cpu"), None)
    all_results = base_results + sweep_results
    if cpu_tp and cpu_tp > 0:
        for r in all_results:
            r.speedup = round(cpu_tp / r.mean_sec / (cpu_tp / (r.num_chunks / cpu_tp * cpu_tp / r.num_chunks)) if r.mean_sec > 0 else 0, 1)
            # Simpler: speedup = r.mean_throughput / cpu_tp
            r.speedup = round(r.mean_throughput / cpu_tp, 1) if cpu_tp > 0 else 1.0

    gpu_name = "N/A"
    gpu_vram_total = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"

    report = f"""# Benchmark Results — Kaizen RAG Embedding Pipeline

## System Info

| Component | Value |
|-----------|-------|
| **GPU** | {gpu_name} |
| **GPU VRAM** | {gpu_vram_total} |
| **Embedding Model** | `{EMBED_MODEL}` |
| **Chunk Size** | {CHUNK_SIZE} chars |
| **Chunks Tested** | {num_chunks:,} |
| **Runs per Config** | {NUM_RUNS} |
| **PyTorch** | {torch.__version__} |
| **CUDA** | {torch.version.cuda if torch.cuda.is_available() else 'N/A'} |

## Core Results — CPU vs GPU vs Precision

| Config | Batch | Throughput (mean ± std) | Per 10K | Peak VRAM | Speedup |
|--------|-------|------------------------|---------|-----------|---------|
"""
    for r in base_results:
        report += (
            f"| {r.config} | {r.batch_size} | "
            f"**{r.mean_throughput:.0f}** ± {r.std_throughput:.0f} chunks/s | "
            f"{r.sec_per_10k:.1f}s | {r.peak_vram_mb:.0f} MB | **{r.speedup}x** |\n"
        )

    # FP16 sweep
    if sweep_results:
        report += f"""
## FP16 Batch Size Sweep — Finding the Sweet Spot

| Batch Size | Throughput (mean ± std) | Per 10K | Peak VRAM | vs CPU |
|-----------|------------------------|---------|-----------|--------|
"""
        for r in sweep_results:
            report += (
                f"| {r.batch_size} | "
                f"**{r.mean_throughput:.0f}** ± {r.std_throughput:.0f} chunks/s | "
                f"{r.sec_per_10k:.1f}s | {r.peak_vram_mb:.0f} MB | **{r.speedup}x** |\n"
            )

    # Key observations
    report += "\n## Key Observations\n\n"

    cpu_r = next((r for r in base_results if r.device == "cpu"), None)
    fp32_r = next((r for r in base_results if r.precision == "fp32" and r.device == "cuda"), None)
    fp16_r = next((r for r in base_results if r.precision == "fp16"), None)

    if cpu_r and fp32_r:
        report += f"1. **CPU → GPU FP32**: {fp32_r.speedup}x speedup ({cpu_r.mean_throughput:.0f} → {fp32_r.mean_throughput:.0f} chunks/s)\n"

    if fp32_r and fp16_r:
        vram_pct = ((fp32_r.peak_vram_mb - fp16_r.peak_vram_mb) / fp32_r.peak_vram_mb * 100) if fp32_r.peak_vram_mb > 0 else 0
        tp_pct = ((fp16_r.mean_throughput - fp32_r.mean_throughput) / fp32_r.mean_throughput * 100) if fp32_r.mean_throughput > 0 else 0
        report += f"2. **FP32 → FP16**: +{tp_pct:.0f}% throughput, {vram_pct:.0f}% less VRAM\n"

    if sweep_results:
        best_sweep = max(sweep_results, key=lambda r: r.mean_throughput)
        report += f"3. **Optimal FP16 batch size**: {best_sweep.batch_size} ({best_sweep.mean_throughput:.0f} chunks/s)\n"

        # Sweet spot analysis
        sorted_sweep = sorted(sweep_results, key=lambda r: r.batch_size)
        report += "\n### Batch Size vs Throughput Trend\n\n"
        report += "```\n"
        max_tp = max(r.mean_throughput for r in sorted_sweep)
        for r in sorted_sweep:
            bar_len = int(r.mean_throughput / max_tp * 40)
            bar = "█" * bar_len
            marker = " ← optimal" if r.batch_size == best_sweep.batch_size else ""
            report += f"  batch {r.batch_size:>5}  |{bar}| {r.mean_throughput:.0f} chunks/s{marker}\n"
        report += "```\n"

    report += f"""
## Methodology

- All configs use identical chunk data from the knowledge base
- **Warmup**: {WARMUP_CHUNKS} chunks discarded before timing (CUDA init)
- **Repetitions**: {NUM_RUNS} timed runs per config (mean ± std reported)
- **Timing**: `torch.cuda.synchronize()` for accurate GPU timing
- **Memory**: Peak VRAM via `torch.cuda.max_memory_allocated()`
- **Isolation**: Full GPU memory cleanup between configs

---
*Generated by `benchmark.py` — Kaizen RAG GPU Optimization Pipeline*
"""
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kaizen RAG Embedding Benchmark")
    parser.add_argument("--chunks", type=int, default=10000, help="Number of chunks")
    parser.add_argument("--quick", action="store_true", help="Quick test (1K chunks)")
    args = parser.parse_args()

    num_chunks = 1000 if args.quick else args.chunks

    print("=" * 60)
    print("  KAIZEN RAG — EMBEDDING BENCHMARK")
    print("=" * 60)
    print(f"  Model:      {EMBED_MODEL}")
    print(f"  Chunks:     {num_chunks:,}")
    print(f"  Runs/cfg:   {NUM_RUNS}")
    print(f"  CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # Load data
    chunks = load_sample_chunks(num_chunks)
    num_chunks = len(chunks)

    # ── Part 1: Base configs ────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PART 1: CORE BENCHMARKS")
    print(f"{'='*60}")

    base_results = []
    for cfg in BASE_CONFIGS:
        result = run_benchmark(cfg, chunks)
        if result:
            base_results.append(result)

    # ── Part 2: FP16 batch sweep ────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  PART 2: FP16 BATCH SIZE SWEEP")
    print(f"{'='*60}")

    sweep_results = []
    for bs in FP16_BATCH_SWEEP:
        cfg = BenchmarkConfig(f"FP16 batch={bs}", "cuda", "fp16", bs)
        result = run_benchmark(cfg, chunks)
        if result:
            sweep_results.append(result)

    # ── Generate report ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("  GENERATING REPORT")
    print(f"{'='*60}")

    report = generate_report(base_results, sweep_results, num_chunks)
    output_path = Path("BENCHMARK_RESULTS.md")
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  📄 Report: {output_path.absolute()}")

    # Raw JSON
    json_path = Path("benchmark_raw.json")
    all_results = base_results + sweep_results
    json_path.write_text(
        json.dumps([asdict(r) for r in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"  📊 Raw data: {json_path.absolute()}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  {r.config:<20} {r.mean_throughput:>7.0f} ± {r.std_throughput:>4.0f} chunks/s  {r.speedup:>5.1f}x  VRAM: {r.peak_vram_mb:>6.0f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
