# Roadmap

## Current State: V1 (Foundation)

Kaizen v1 is the **retrieval and ranking engine**. It handles ingestion, embedding, search, reranking, and LLM response generation. The frontend application lives in a separate project and will consume this engine.

---

## Short Term

- **Connect frontend application** to the Kaizen RAG backend (API layer between frontend and retrieval engine)
- **User testing** with real queries against the full knowledge base
- **Improve PDF chunking** to preserve table structure and section headers
- **Expand evaluation dataset** beyond 23 queries to cover more edge cases

## Mid Term

- **API layer** (FastAPI or similar) exposing `/query`, `/ingest`, `/metrics` endpoints for the frontend
- **Multi-index retrieval** if corpus grows and single-index quality degrades per domain
- **Hybrid search** combining BM25 keyword matching with embedding similarity
- **Reranker tuning** to address the 2 documented regressions (medallion architecture, fine-tuning)
- **Semantic chunking** that respects document structure instead of fixed character windows

## Long Term

- **Code index** with AST-aware chunking for source code files
- **Agent tooling** for multi-step reasoning over retrieved context
- **Streaming ingestion** for incremental document updates without full re-index
- **Multi-model evaluation** comparing embedding models beyond `all-MiniLM-L6-v2`
