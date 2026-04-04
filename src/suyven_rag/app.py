"""Suyven v1 — RAG web interface (Streamlit)."""

from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from suyven_rag.rag.config import KNOWLEDGE_DIR, WORKERS
from suyven_rag.rag.index_registry import get_index, reset_index
from suyven_rag.rag.llm import stream_chat
from suyven_rag.rag.loader import iter_files
from suyven_rag.rag.model_registry import get_embed_model
from suyven_rag.rag.monitoring import gpu_metrics
from suyven_rag.rag.orchestrator import execute_search, format_context, plan
from suyven_rag.rag.pipeline import read_and_chunk
from suyven_rag.rag.store import add_chunks, ensure_ollama

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Suyven v1",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Init (runs once per session)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Starting Ollama...")
def init():
    ensure_ollama()
    col = get_index()
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
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚡ Suyven v1")
    st.caption("Personal knowledge RAG")
    st.divider()

    # ── GPU metrics dashboard ──────────────────────────────────────────
    gpu = gpu_metrics()
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
    force = st.checkbox(
        "Full re-index", value=False, help="Wipe the collection and re-index everything"
    )

    if st.button("🚀 Ingest Now", use_container_width=True):
        files = list(iter_files(KNOWLEDGE_DIR))
        if not files:
            st.error(f"No files found in `{KNOWLEDGE_DIR}`")
        else:
            target_col = reset_index() if force else col

            # ── Phase 1: Parallel read + chunk ──────────────────────
            phase1 = st.progress(0, text="📖 Reading & chunking files…")
            file_chunks: list[tuple] = []
            total_chunk_count = 0

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(read_and_chunk, f): i for i, f in enumerate(files)}
                for done, future in enumerate(futures, 1):
                    result = future.result()
                    file_chunks.append(result)
                    total_chunk_count += len(result[1])
                    phase1.progress(
                        done / len(files),
                        text=f"� Read {done}/{len(files)} files — {total_chunk_count} chunks found",
                    )

            phase1.progress(
                1.0, text=f"📖 Done! {total_chunk_count} chunks from {len(files)} files"
            )

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
                status.caption(f"✅ {total_added} chunks added · ⏭️ {total_skipped} skipped")

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

st.title("Ask your knowledge base")

query = st.text_input(
    label="Query",
    placeholder="What is a star schema?",
    label_visibility="collapsed",
)

if query:
    with st.spinner("🔍 Searching knowledge base..."):
        route = plan(query, category=category, top_k=n_results)
        results = execute_search(query, route, category=category)

    if not results:
        st.warning("No results found.")
    else:
        # Build context from retrieved chunks
        context = format_context(results)

        # Stream LLM response
        st.subheader("💡 Answer")

        st.write_stream(stream_chat(query, context))

        # Show source chunks as collapsible references
        st.divider()
        st.caption(f"📚 {len(results)} sources used")
        for r in results:
            path = f"{r['category']}/{r['source']}"
            with st.expander(f"**{path}** — {r['score']:.0%} relevance"):
                st.markdown(r["text"])
