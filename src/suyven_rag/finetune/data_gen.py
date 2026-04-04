"""Generate synthetic (query, passage) training pairs from the ChromaDB corpus.

For each sampled chunk, calls Groq to generate plausible questions a user
would ask that the chunk answers. Output: JSONL with {query, positive, source, category}.

Usage:
    python -m finetune.data_gen
    python -m finetune.data_gen --samples 500 --questions 3
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import requests

from suyven_rag.finetune.config import TrainConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

QUESTION_PROMPT = """Given this technical passage, generate {n} specific questions that this passage would answer well.
Return only the questions, one per line. No numbering, no bullets, no extra text.

Passage:
{text}"""


def sample_chunks(n: int) -> list[dict]:
    """Sample n chunks from ChromaDB, stratified by category."""
    from suyven_rag.rag.index_registry import get_index

    col = get_index()
    total = col.count()
    if total == 0:
        raise RuntimeError("ChromaDB is empty. Run ingestion first.")

    # Get all chunks (ChromaDB doesn't support random sampling natively)
    # Fetch in batches to avoid memory issues
    batch_size = 5000
    all_docs = []
    for offset in range(0, total, batch_size):
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(result["documents"], result["metadatas"], strict=False):
            all_docs.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "category": meta.get("category", ""),
                }
            )

    logger.info("Loaded %d chunks from ChromaDB", len(all_docs))

    # Stratified sampling: proportional to category size
    by_cat: dict[str, list[dict]] = {}
    for doc in all_docs:
        cat = doc["category"]
        by_cat.setdefault(cat, []).append(doc)

    sampled = []
    for _cat, docs in by_cat.items():
        proportion = len(docs) / len(all_docs)
        cat_n = max(1, int(n * proportion))
        sampled.extend(random.sample(docs, min(cat_n, len(docs))))

    random.shuffle(sampled)
    return sampled[:n]


def generate_questions(
    text: str,
    n_questions: int,
    api_url: str,
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    """Call Groq to generate questions for a chunk, with backoff on 429."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    prompt = QUESTION_PROMPT.format(n=n_questions, text=text[:1500])

    for attempt in range(max_retries):
        resp = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 2**attempt * 5  # 5s, 10s, 20s
            logger.debug("Rate limited, waiting %ds...", wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()

        content = resp.json()["choices"][0]["message"]["content"]
        questions = [q.strip() for q in content.strip().split("\n") if q.strip()]
        return questions[:n_questions]

    raise requests.exceptions.HTTPError("Rate limited after max retries")


def run(config: TrainConfig) -> Path:
    """Generate training pairs and save to JSONL."""
    from suyven_rag.rag.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

    output = config.train_data_path
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling %d chunks from corpus...", config.sample_chunks)
    chunks = sample_chunks(config.sample_chunks)
    logger.info("Sampled %d chunks across categories", len(chunks))

    pairs = []
    errors = 0
    batch_size = config.groq_batch_size

    for i, chunk in enumerate(chunks):
        try:
            questions = generate_questions(
                text=chunk["text"],
                n_questions=config.questions_per_chunk,
                api_url=LLM_API_URL,
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
            )
            for q in questions:
                pairs.append(
                    {
                        "query": q,
                        "positive": chunk["text"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                    }
                )
        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.warning("Error on chunk %d: %s", i, e)
            if errors == 5:
                logger.warning("Suppressing further error logs...")

        # Rate limiting
        if (i + 1) % batch_size == 0:
            logger.info(
                "Progress: %d/%d chunks, %d pairs generated, %d errors",
                i + 1,
                len(chunks),
                len(pairs),
                errors,
            )
            time.sleep(config.groq_delay_s)

    # Save
    with open(output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(
        "Done: %d pairs from %d chunks (%d errors). Saved to %s",
        len(pairs),
        len(chunks),
        errors,
        output,
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate training pairs from corpus")
    parser.add_argument("--samples", type=int, default=3000, help="Number of chunks to sample")
    parser.add_argument("--questions", type=int, default=2, help="Questions per chunk")
    args = parser.parse_args()

    config = TrainConfig(sample_chunks=args.samples, questions_per_chunk=args.questions)
    run(config)


if __name__ == "__main__":
    main()
