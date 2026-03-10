"""Kaizen v1 — RAG web interface (Streamlit)."""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor

from rag.chunker import chunk_text
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR, OLLAMA_URL
from rag.loader import iter_files, read_file
from rag.retriever import format_context, search
from rag.store import (
    add_chunks,
    ensure_ollama,
    get_collection,
    get_embed_model,
    reset_collection,
)

# ---------------------------------------------------------------------------
# GPU monitoring helpers (pynvml)
# ---------------------------------------------------------------------------

_PYNVML_OK = False
try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    pass


def _gpu_metrics() -> dict | None:
    """Return GPU metrics dict or None if pynvml unavailable."""
    if not _PYNVML_OK:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {
            "name": name,
            "vram_used_gb": mem.used / (1024 ** 3),
            "vram_total_gb": mem.total / (1024 ** 3),
            "vram_pct": mem.used / mem.total * 100,
            "gpu_util": util.gpu,
            "mem_util": util.memory,
            "temp_c": temp,
        }
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Kaizen v1",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Init (runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Starting Ollama...")
def init():
    ensure_ollama()
    col = get_collection()
    return col


col = init()
try:
    count = col.count()
except Exception:
    # Collection was deleted (e.g. failed re-index) — recreate
    st.cache_resource.clear()
    col = init()
    count = col.count()


# ---------------------------------------------------------------------------
# Helpers — parallel read + chunk
# ---------------------------------------------------------------------------

def _read_and_chunk(path):
    """Read a file and return its chunks + metadata (runs in thread)."""
    text = read_file(path)
    if not text.strip():
        return path, []
    return path, chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ Kaizen v1")
    st.caption("Personal knowledge RAG")
    st.divider()

    # ── GPU metrics dashboard ──────────────────────────────────────────
    gpu = _gpu_metrics()
    if gpu:
        st.caption(f"🖥️ **{gpu['name']}**")
        col1, col2 = st.columns(2)
        col1.metric("🌡️ Temp", f"{gpu['temp_c']}°C")
        col2.metric("⚙️ GPU", f"{gpu['gpu_util']}%")

        vram_label = f"{gpu['vram_used_gb']:.1f} / {gpu['vram_total_gb']:.1f} GB"
        st.progress(gpu["vram_pct"] / 100, text=f"VRAM: {vram_label}")
        st.divider()

    st.metric("Chunks indexed", count)

    # ── Ingest section ────────────────────────────────────────────────
    force = st.checkbox("Full re-index", value=False,
                        help="Wipe the collection and re-index everything")

    if st.button("🚀 Ingest Now", use_container_width=True):
        files = list(iter_files(KNOWLEDGE_DIR))
        if not files:
            st.error(f"No files found in `{KNOWLEDGE_DIR}`")
        else:
            target_col = reset_collection() if force else col

            # ── Phase 1: Parallel read + chunk ──────────────────────
            phase1 = st.progress(0, text="📖 Reading & chunking files…")
            file_chunks: list[tuple] = []
            total_chunk_count = 0

            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_read_and_chunk, f): i for i, f in enumerate(files)}
                done = 0
                for future in futures:
                    result = future.result()
                    file_chunks.append(result)
                    total_chunk_count += len(result[1])
                    done += 1
                    phase1.progress(
                        done / len(files),
                        text=f"� Read {done}/{len(files)} files — {total_chunk_count} chunks found",
                    )

            phase1.progress(1.0, text=f"📖 Done! {total_chunk_count} chunks from {len(files)} files")

            # ── Phase 2: Pre-load embedding model ───────────────────
            with st.spinner("🧠 Loading embedding model to GPU…"):
                get_embed_model()

            # ── Phase 3: Embed + add to ChromaDB ────────────────────
            phase2 = st.progress(0, text="⚡ Embedding & indexing…")
            status = st.empty()
            total_added = total_skipped = processed = 0

            for path, chunks in file_chunks:
                if not chunks:
                    processed += 1
                    continue

                added, skipped = add_chunks(target_col, path, chunks, KNOWLEDGE_DIR)
                total_added += added
                total_skipped += skipped
                processed += 1

                rel = str(path.relative_to(KNOWLEDGE_DIR))
                phase2.progress(
                    processed / len(file_chunks),
                    text=f"⚡ {processed}/{len(file_chunks)} — {rel}",
                )
                status.caption(
                    f"✅ {total_added} chunks added · ⏭️ {total_skipped} skipped"
                )

            phase2.progress(1.0, text="✅ Ingestion complete!")
            st.success(
                f"Done! **{len(files)}** files → "
                f"**{total_added}** chunks added, "
                f"**{total_skipped}** skipped. "
                f"Total in DB: **{target_col.count()}**"
            )
            st.cache_resource.clear()

    if count == 0:
        st.warning("Knowledge base empty — click **Ingest Now** above.")

    st.divider()

    n_results = st.slider("Results", min_value=1, max_value=15, value=5)

    category = st.text_input(
        "Filter by category",
        placeholder="e.g. ai, data-engineering",
        help="Leave empty to search all categories",
    )
    category = category.strip() or None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

LLM_MODEL = "qwen3:8b"

SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer the user's question using ONLY the provided context from their knowledge base. 
- Synthesize information from all relevant sources into a clear, complete answer.
- Use markdown formatting for readability.
- If the context doesn't contain enough information, say so honestly.
- Cite sources using [source_name] when referencing specific documents.
- Answer in the same language as the question."""

st.title("Ask your knowledge base")

query = st.text_input(
    label="Query",
    placeholder="What is a star schema?",
    label_visibility="collapsed",
)

if query:
    with st.spinner("🔍 Searching knowledge base..."):
        results = search(query, n=n_results, category=category)

    if not results:
        st.warning("No results found.")
    else:
        # Build context from retrieved chunks
        context = format_context(results)

        # Stream LLM response
        st.subheader("💡 Answer")

        def stream_response():
            import requests as req
            resp = req.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                    ],
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

        st.write_stream(stream_response())

        # Show source chunks as collapsible references
        st.divider()
        st.caption(f"📚 {len(results)} sources used")
        for r in results:
            path = f"{r['category']}/{r['source']}"
            with st.expander(f"**{path}** — {r['score']:.0%} relevance"):
                st.markdown(r["text"])

