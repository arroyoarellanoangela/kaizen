"""Self-supervised training pair generation — no LLM needed.

Strategies:
  1. Same-document pairs: two chunks from the same source = positive pair
     (they share topic/context, model learns document coherence)
  2. Title-as-query: source filename as pseudo-query, chunk as positive
     (model learns to match topic labels to content)
  3. Adjacent chunks: consecutive chunks from same source = positive pair
     (model learns local coherence, paragraph continuity)

Hard negatives:
  For each positive pair, find chunks that are SIMILAR (high embedding score)
  but from a DIFFERENT source. These are the confusing cases the model needs
  to learn to distinguish.

Usage:
    python -m finetune.data_gen_selfsup
    python -m finetune.data_gen_selfsup --pairs 10000 --hard-negatives
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "finetune" / "pairs_selfsup.jsonl"


def load_corpus() -> dict[str, list[dict]]:
    """Load all chunks from ChromaDB, grouped by source."""
    from suyven_rag.rag.index_registry import get_index

    col = get_index()
    total = col.count()
    logger.info("Loading %d chunks from ChromaDB...", total)

    by_source: dict[str, list[dict]] = defaultdict(list)
    batch_size = 5000

    for offset in range(0, total, batch_size):
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(result["documents"], result["metadatas"], strict=False):
            source = meta.get("source", "unknown")
            chunk_idx = int(meta.get("chunk_index", "0"))
            by_source[source].append(
                {
                    "text": doc,
                    "source": source,
                    "category": meta.get("category", ""),
                    "chunk_index": chunk_idx,
                }
            )

    # Sort each source's chunks by index for adjacency pairs
    for source in by_source:
        by_source[source].sort(key=lambda x: x["chunk_index"])

    logger.info("Loaded %d sources, %d total chunks", len(by_source), total)
    return dict(by_source)


def generate_same_document_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Strategy 1: Random pairs of chunks from the same document.

    Two chunks sharing a source discuss the same topic, so they're positive pairs.
    """
    rng = random.Random(seed)
    pairs = []

    sources = [s for s, chunks in by_source.items() if len(chunks) >= 2]
    rng.shuffle(sources)

    for source in sources:
        chunks = by_source[source]
        if len(chunks) < 2:
            continue

        # Sample pairs from this source (proportional to size, but capped)
        n_pairs = min(len(chunks) // 2, max(1, max_pairs // len(sources)))

        for _ in range(n_pairs):
            a, b = rng.sample(range(len(chunks)), 2)
            pairs.append(
                {
                    "query": chunks[a]["text"],
                    "positive": chunks[b]["text"],
                    "source": source,
                    "category": chunks[a]["category"],
                    "strategy": "same_document",
                }
            )
            if len(pairs) >= max_pairs:
                return pairs

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def generate_adjacent_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Strategy 2: Adjacent chunks (i, i+1) from the same document.

    Consecutive chunks have strong topical coherence — the model learns
    that nearby text in a document is semantically related.
    """
    rng = random.Random(seed)
    pairs = []

    for source, chunks in by_source.items():
        for i in range(len(chunks) - 1):
            pairs.append(
                {
                    "query": chunks[i]["text"],
                    "positive": chunks[i + 1]["text"],
                    "source": source,
                    "category": chunks[i]["category"],
                    "strategy": "adjacent",
                }
            )

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def generate_title_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 3000,
    seed: int = 42,
) -> list[dict]:
    """Strategy 3: Source name as pseudo-query, chunk as positive.

    The source filename is a rough topic label. Teaching the model to
    associate topic names with their content.
    """
    rng = random.Random(seed)
    pairs = []

    for source, chunks in by_source.items():
        # Clean up source name to make a pseudo-query
        title = source.replace("_", " ").replace("-", " ").strip()
        if len(title) < 3:
            continue

        # Sample chunks from this source
        n = min(len(chunks), max(1, max_pairs // len(by_source)))
        sampled = rng.sample(chunks, min(n, len(chunks)))

        for chunk in sampled:
            pairs.append(
                {
                    "query": title,
                    "positive": chunk["text"],
                    "source": source,
                    "category": chunk["category"],
                    "strategy": "title_as_query",
                }
            )

    rng.shuffle(pairs)
    return pairs[:max_pairs]


def mine_hard_negatives(
    pairs: list[dict],
    by_source: dict[str, list[dict]],
    top_k: int = 10,
    max_pairs: int = 5000,
) -> list[dict]:
    """Add hard negatives: chunks that are similar but from different sources.

    For each (query, positive) pair, embed the query, find similar chunks
    from OTHER sources. These become hard negatives that teach the model
    to distinguish between superficially similar but topically different content.
    """
    from suyven_rag.rag.store import embed_batch

    logger.info("Mining hard negatives for %d pairs...", len(pairs))

    # Build a pool of chunks from all sources with their embeddings
    all_chunks = []
    for _source, chunks in by_source.items():
        # Sample to keep pool manageable
        sampled = random.sample(chunks, min(200, len(chunks)))
        all_chunks.extend(sampled)

    logger.info("Embedding %d pool chunks...", len(all_chunks))
    pool_texts = [c["text"] for c in all_chunks]

    # Embed in batches
    import numpy as np

    pool_embeds = np.array(embed_batch(pool_texts))

    # For each pair, find hard negatives
    triplets = []
    batch_size = 256

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        queries = [p["query"] for p in batch]
        query_embeds = np.array(embed_batch(queries))

        # Cosine similarity
        sims = query_embeds @ pool_embeds.T  # (batch, pool)

        for j, pair in enumerate(batch):
            source = pair["source"]
            # Get top-k similar chunks from DIFFERENT sources
            ranked = np.argsort(sims[j])[::-1]

            for idx in ranked[: top_k * 3]:  # search wider to find enough cross-source
                candidate = all_chunks[idx]
                if candidate["source"] != source:
                    triplets.append(
                        {
                            "query": pair["query"],
                            "positive": pair["positive"],
                            "negative": candidate["text"],
                            "source": source,
                            "category": pair["category"],
                            "neg_source": candidate["source"],
                            "strategy": pair["strategy"] + "_hard_neg",
                        }
                    )
                    break

        if (i + batch_size) % 1000 == 0:
            logger.info(
                "Hard neg mining: %d/%d done, %d triplets",
                i + batch_size,
                len(pairs),
                len(triplets),
            )

    logger.info("Mined %d hard negative triplets", len(triplets))
    return triplets[:max_pairs]


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove duplicate pairs by hashing query+positive."""
    seen = set()
    unique = []
    for p in pairs:
        h = hashlib.md5((p["query"][:200] + p["positive"][:200]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)
    return unique


def run(
    target_pairs: int = 10000,
    hard_negatives: bool = False,
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
) -> Path:
    """Generate self-supervised training pairs."""
    by_source = load_corpus()

    # Strategy allocation: 40% same-doc, 40% adjacent, 20% title
    n_same = int(target_pairs * 0.4)
    n_adjacent = int(target_pairs * 0.4)
    n_title = int(target_pairs * 0.2)

    logger.info("Generating pairs: %d same-doc, %d adjacent, %d title", n_same, n_adjacent, n_title)

    same_doc = generate_same_document_pairs(by_source, max_pairs=n_same, seed=seed)
    adjacent = generate_adjacent_pairs(by_source, max_pairs=n_adjacent, seed=seed)
    title = generate_title_pairs(by_source, max_pairs=n_title, seed=seed)

    all_pairs = same_doc + adjacent + title
    all_pairs = deduplicate(all_pairs)
    random.Random(seed).shuffle(all_pairs)

    logger.info(
        "Generated %d unique pairs (same_doc=%d, adjacent=%d, title=%d)",
        len(all_pairs),
        len(same_doc),
        len(adjacent),
        len(title),
    )

    # Hard negative mining
    triplets = []
    if hard_negatives:
        triplets = mine_hard_negatives(all_pairs[:5000], by_source, max_pairs=5000)

    # Save pairs (query + positive format, compatible with existing dataset.py)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(
                json.dumps(
                    {
                        "query": p["query"],
                        "positive": p["positive"],
                        "source": p["source"],
                        "category": p["category"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info("Saved %d pairs to %s", len(all_pairs), output)

    # Save triplets separately if generated
    if triplets:
        triplet_path = output.parent / "triplets.jsonl"
        with open(triplet_path, "w", encoding="utf-8") as f:
            for t in triplets:
                f.write(
                    json.dumps(
                        {
                            "query": t["query"],
                            "positive": t["positive"],
                            "negative": t["negative"],
                            "source": t["source"],
                            "category": t["category"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info("Saved %d triplets to %s", len(triplets), triplet_path)

    return output


def main():
    parser = argparse.ArgumentParser(description="Self-supervised training pair generation")
    parser.add_argument("--pairs", type=int, default=10000, help="Target number of pairs")
    parser.add_argument("--hard-negatives", action="store_true", help="Mine hard negatives")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        target_pairs=args.pairs,
        hard_negatives=args.hard_negatives,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
