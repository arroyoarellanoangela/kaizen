"""V2 training data generation — smarter self-supervised + cross-encoder filtering.

Problems with V1 self-supervised:
  - "Two chunks from same doc" is weak signal (they share topic, not query intent)
  - Title-as-query is too vague ("aws lambda" doesn't match specific chunk content)

V2 strategies:
  1. First-sentence-as-query: extract opening sentence from chunk, pair with full chunk
     (opening sentences often summarize what follows — natural pseudo-queries)
  2. Key-phrase extraction: pull noun phrases / definitions as pseudo-queries
  3. Cross-encoder FILTERING: score every pair with the reranker, discard low scores
     (the reranker knows what a real query-passage match looks like)
  4. Mix with existing Groq-generated pairs (highest quality anchor)

Usage:
    python -m finetune.data_gen_v2
    python -m finetune.data_gen_v2 --target 3000 --min-score 0.3
"""

import argparse
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR / "data" / "finetune" / "pairs_v2.jsonl"
GROQ_PAIRS = BASE_DIR / "data" / "finetune" / "pairs.jsonl"


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

    for source in by_source:
        by_source[source].sort(key=lambda x: x["chunk_index"])

    logger.info("Loaded %d sources, %d total chunks", len(by_source), total)
    return dict(by_source)


# ---------------------------------------------------------------------------
# Strategy 1: First sentence as pseudo-query
# ---------------------------------------------------------------------------


def extract_first_sentence(text: str) -> str | None:
    """Extract the first meaningful sentence from a chunk."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        s = s.strip()
        # Skip very short or code-like lines
        if len(s) < 20 or len(s) > 200:
            continue
        # Skip lines that are mostly code/symbols
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.6:
            continue
        return s
    return None


def generate_first_sentence_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 10000,
    seed: int = 42,
) -> list[dict]:
    """Use opening sentence of each chunk as pseudo-query for that chunk."""
    rng = random.Random(seed)
    pairs = []

    all_chunks = [c for chunks in by_source.values() for c in chunks]
    rng.shuffle(all_chunks)

    for chunk in all_chunks:
        sentence = extract_first_sentence(chunk["text"])
        if not sentence:
            continue
        # The query is the first sentence, the positive is the full chunk
        # (but we remove the first sentence from the positive to avoid trivial matching)
        remaining = chunk["text"][len(sentence) :].strip()
        if len(remaining) < 50:
            continue

        pairs.append(
            {
                "query": sentence,
                "positive": remaining,
                "source": chunk["source"],
                "category": chunk["category"],
                "strategy": "first_sentence",
            }
        )
        if len(pairs) >= max_pairs:
            break

    logger.info("First-sentence pairs: %d", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Strategy 2: Definition/heading as pseudo-query
# ---------------------------------------------------------------------------

DEFINITION_PATTERNS = [
    r"^([A-Z][A-Za-z\s]{5,40})\s+(?:is|are|refers to|means)\s+",  # "X is ..."
    r"^(?:What is|How to|Why does|When to)\s+.{10,80}",  # Question-like openings
    r"^#+\s*(.{5,60})",  # Markdown headings
]


def generate_definition_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Extract definitions and headings as pseudo-queries."""
    rng = random.Random(seed)
    pairs = []

    all_chunks = [c for chunks in by_source.values() for c in chunks]
    rng.shuffle(all_chunks)

    for chunk in all_chunks:
        text = chunk["text"].strip()
        for pattern in DEFINITION_PATTERNS:
            m = re.match(pattern, text, re.MULTILINE)
            if m:
                query = m.group(0).strip().rstrip(":").strip()
                if 10 <= len(query) <= 100:
                    pairs.append(
                        {
                            "query": query,
                            "positive": text,
                            "source": chunk["source"],
                            "category": chunk["category"],
                            "strategy": "definition",
                        }
                    )
                    break
        if len(pairs) >= max_pairs:
            break

    logger.info("Definition pairs: %d", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Strategy 3: Question reformulation from chunk content
# ---------------------------------------------------------------------------


def generate_question_pairs(
    by_source: dict[str, list[dict]],
    max_pairs: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """Turn declarative statements into questions.

    "Lambda functions support Python 3.12" -> "What languages does Lambda support?"
    This is a cheap heuristic, not LLM-quality, but better than raw chunk pairs.
    """
    rng = random.Random(seed)
    pairs = []

    # Simple templates: extract subject and create "What is X?" style queries
    all_chunks = [c for chunks in by_source.values() for c in chunks]
    rng.shuffle(all_chunks)

    for chunk in all_chunks:
        text = chunk["text"].strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for s in sentences[:3]:  # Only look at first 3 sentences
            s = s.strip()
            if len(s) < 30 or len(s) > 150:
                continue

            # "X is/are Y" -> "What is X?"
            m = re.match(r"^([A-Z][A-Za-z\s\-]{3,40})\s+(?:is|are)\s+", s)
            if m:
                subject = m.group(1).strip()
                pairs.append(
                    {
                        "query": f"What is {subject}?",
                        "positive": text,
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "strategy": "question_reformat",
                    }
                )
                break

            # "To X, you need Y" -> "How to X?"
            m = re.match(r"^To\s+(.{10,60}),", s)
            if m:
                action = m.group(1).strip()
                pairs.append(
                    {
                        "query": f"How to {action}?",
                        "positive": text,
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "strategy": "question_reformat",
                    }
                )
                break

        if len(pairs) >= max_pairs:
            break

    logger.info("Question-reformat pairs: %d", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Cross-encoder filtering
# ---------------------------------------------------------------------------


def filter_with_reranker(
    pairs: list[dict],
    min_score: float = 0.3,
    batch_size: int = 64,
) -> list[dict]:
    """Score pairs with the cross-encoder reranker and filter low scores.

    The reranker is trained on real query-passage pairs (MS MARCO), so it
    knows what a genuine match looks like. Pairs that score below min_score
    are likely noise.
    """
    from suyven_rag.rag.model_registry import get_reranker

    reranker = get_reranker()
    logger.info("Filtering %d pairs with cross-encoder (min_score=%.2f)...", len(pairs), min_score)

    scored = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = [(p["query"], p["positive"]) for p in batch]
        scores = reranker.predict(inputs, show_progress_bar=False)

        for pair, score in zip(batch, scores, strict=False):
            pair["reranker_score"] = float(score)
            scored.append(pair)

        if (i + batch_size) % 1000 == 0:
            logger.info("Scored %d/%d pairs...", min(i + batch_size, len(pairs)), len(pairs))

    # Filter
    kept = [p for p in scored if p["reranker_score"] >= min_score]
    # Sort by score descending (best pairs first)
    kept.sort(key=lambda x: x["reranker_score"], reverse=True)

    avg_score = np.mean([p["reranker_score"] for p in scored]) if scored else 0
    avg_kept = np.mean([p["reranker_score"] for p in kept]) if kept else 0
    logger.info(
        "Reranker filter: %d/%d kept (%.1f%%), avg_score=%.3f, avg_kept=%.3f",
        len(kept),
        len(scored),
        100 * len(kept) / max(len(scored), 1),
        avg_score,
        avg_kept,
    )

    return kept


# ---------------------------------------------------------------------------
# Load existing Groq pairs
# ---------------------------------------------------------------------------


def load_groq_pairs(path: Path = GROQ_PAIRS) -> list[dict]:
    """Load existing LLM-generated pairs (highest quality anchor)."""
    if not path.exists():
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entry["strategy"] = "groq_llm"
            entry["reranker_score"] = 1.0  # Assume LLM pairs are high quality
            pairs.append(entry)
    logger.info("Loaded %d existing Groq LLM pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    target_pairs: int = 3000,
    min_score: float = 0.3,
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
) -> Path:
    """Generate high-quality training pairs with cross-encoder filtering."""
    by_source = load_corpus()

    # Generate candidates from all strategies
    first_sent = generate_first_sentence_pairs(by_source, max_pairs=target_pairs * 2, seed=seed)
    definitions = generate_definition_pairs(by_source, max_pairs=target_pairs, seed=seed)
    questions = generate_question_pairs(by_source, max_pairs=target_pairs, seed=seed)

    all_candidates = first_sent + definitions + questions
    logger.info("Total candidates before filtering: %d", len(all_candidates))

    # Cross-encoder filtering (the key quality gate)
    filtered = filter_with_reranker(all_candidates, min_score=min_score)

    # Add Groq pairs (always included, they're the gold standard)
    groq = load_groq_pairs()

    # Combine: Groq first (high quality), then filtered self-supervised
    combined = groq + filtered[: target_pairs - len(groq)]

    # Deduplicate
    import hashlib

    seen = set()
    unique = []
    for p in combined:
        h = hashlib.md5((p["query"][:200] + p["positive"][:200]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)

    random.Random(seed).shuffle(unique)

    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p in unique:
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

    # Stats
    strategy_counts = defaultdict(int)
    for p in unique:
        strategy_counts[p.get("strategy", "unknown")] += 1

    logger.info("Final dataset: %d pairs", len(unique))
    for strategy, count in sorted(strategy_counts.items()):
        logger.info("  %s: %d pairs", strategy, count)
    logger.info("Saved to %s", output)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="V2 training data generation with cross-encoder filtering"
    )
    parser.add_argument("--target", type=int, default=3000, help="Target number of pairs")
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="Min reranker score to keep a pair"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        target_pairs=args.target,
        min_score=args.min_score,
        output=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
