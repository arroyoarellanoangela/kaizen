# Known Limitations

## Retrieval

- **Vector database indexing query scores 0.00**: The query "How does vector database indexing work?" fails to match expected sources in both bi-encoder and reranker pipelines. Ground truth keywords may need refinement, or the knowledge base lacks direct coverage.
- **Long queries can mix contexts**: Very specific multi-topic queries (e.g., Spark + S3 + Redshift) occasionally pull chunks from tangentially related documents.
- **Overfetch ratio is fixed at 4x**: No dynamic adjustment based on query complexity or candidate quality distribution.

## Reranker

- **Two regressions documented**: "medallion architecture" (-0.20 NDCG@5) and "fine-tuning techniques" (-0.23 NDCG@5) perform worse with the reranker than bi-encoder alone. The cross-encoder sometimes promotes semantically similar but less relevant chunks.
- **GPU FP32 offers no speedup over CPU** for the reranker at 20 candidates. The batch is too small to amortize GPU transfer overhead. FP16 is required to see a benefit.
- **Latency scales linearly with candidates**: 7 ms at 10 candidates, 25 ms at 60. For large candidate sets (>40), the reranker adds noticeable latency.

## Ingestion

- **PDF table extraction loses structure**: PyMuPDF extracts text linearly, so tabular data in PDFs becomes flat text. Column relationships are lost.
- **No deduplication across files**: If the same content appears in multiple files, it gets indexed multiple times.
- **Fixed chunk size (600 chars)**: No semantic chunking. Long paragraphs get split mid-sentence at the boundary.

## Platform

- **Windows stdout encoding**: Python on Windows defaults to cp1252, which cannot render Unicode box-drawing characters or emoji. Scripts use `sys.stdout.reconfigure(encoding="utf-8")` as workaround.
- **pynvml deprecation warning**: The `pynvml` package shows a deprecation notice in favor of `nvidia-ml-py`. Both provide the same `pynvml` module; only the pip package name differs.
- **Ollama must be running**: The LLM chat requires Ollama with `qwen3:8b` pulled. `ensure_ollama()` auto-starts it, but if the model isn't downloaded, the first query will fail.

## Evaluation

- **Ground truth is manually curated**: The 23 validation queries and their expected sources are hand-picked. Coverage of the full knowledge base is incomplete.
- **Binary relevance only**: A chunk either matches an expected source keyword or it doesn't. No graded relevance (partially relevant = 0).
- **No latency-under-load testing**: All benchmarks run single-query sequential. No concurrent query stress testing.
