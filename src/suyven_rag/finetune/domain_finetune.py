"""Domain-specific embedding fine-tuning — end-to-end pipeline.

Connects the domain registry with the LoRA fine-tuning infrastructure.
When a domain has enough data, this pipeline:

1. Samples chunks from the domain's ChromaDB collection
2. Generates training pairs (self-supervised: first-sentence, definition, question-reformat)
3. Filters pairs with cross-encoder reranker (quality gate)
4. Trains LoRA adapters on the base embed model
5. Merges LoRA into a domain-specific model
6. Registers the model in model_registry for automatic use

Usage:
    python -m finetune.domain_finetune oncologia
    python -m finetune.domain_finetune oncologia --epochs 5 --min-pairs 200
"""

import argparse
import hashlib
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DOMAIN_FT_DIR = BASE_DIR / "data" / "finetune" / "domains"


@dataclass
class DomainFinetuneConfig:
    """Config for domain-specific fine-tuning."""

    slug: str
    # Data gen
    min_pairs: int = 200  # minimum pairs to proceed with training
    target_pairs: int = 2000  # target number of training pairs
    min_reranker_score: float = 0.2  # quality gate for self-supervised pairs
    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    # Training
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32  # smaller batches for domain data (less data)
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    temperature: float = 0.05
    max_seq_length: int = 128
    eval_split: float = 0.1


@dataclass
class DomainFinetuneResult:
    """Result of a domain fine-tune run."""

    slug: str
    status: str  # "success", "insufficient_data", "error"
    pairs_generated: int = 0
    pairs_after_filter: int = 0
    train_pairs: int = 0
    eval_pairs: int = 0
    final_train_loss: float | None = None
    final_eval_loss: float | None = None
    training_time_s: float = 0.0
    model_path: str = ""
    lora_path: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Step 1: Sample chunks from domain collection
# ---------------------------------------------------------------------------


def sample_domain_chunks(slug: str, max_chunks: int = 5000) -> list[dict]:
    """Load chunks from a domain's ChromaDB collection."""
    from suyven_rag.rag.index_registry import get_index

    index_name = f"domain_{slug}"
    col = get_index(index_name)
    total = col.count()

    if total == 0:
        return []

    all_docs = []
    batch_size = 5000
    for offset in range(0, min(total, max_chunks), batch_size):
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

    logger.info("[%s] Loaded %d chunks from domain collection", slug, len(all_docs))
    return all_docs


# ---------------------------------------------------------------------------
# Step 2: Generate training pairs (self-supervised, no LLM calls)
# ---------------------------------------------------------------------------


def _extract_first_sentence(text: str) -> str | None:
    """Extract first meaningful sentence from a chunk."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        s = s.strip()
        if len(s) < 20 or len(s) > 200:
            continue
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.6:
            continue
        return s
    return None


def _generate_first_sentence_pairs(
    chunks: list[dict], max_pairs: int, seed: int = 42
) -> list[dict]:
    """Opening sentence as pseudo-query, rest of chunk as positive."""
    rng = random.Random(seed)
    pairs = []
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    for chunk in shuffled:
        sentence = _extract_first_sentence(chunk["text"])
        if not sentence:
            continue
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
    return pairs


def _generate_definition_pairs(chunks: list[dict], max_pairs: int, seed: int = 42) -> list[dict]:
    """Extract definitions and headings as pseudo-queries."""
    patterns = [
        r"^([A-Z][A-Za-z\s]{5,40})\s+(?:is|are|refers to|means|es|son|se refiere a)\s+",
        r"^(?:What is|How to|Why does|Que es|Como)\s+.{10,80}",
        r"^#+\s*(.{5,60})",
    ]
    rng = random.Random(seed)
    pairs = []
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    for chunk in shuffled:
        text = chunk["text"].strip()
        for pattern in patterns:
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
    return pairs


def _generate_question_pairs(chunks: list[dict], max_pairs: int, seed: int = 42) -> list[dict]:
    """Turn declarative statements into questions."""
    rng = random.Random(seed)
    pairs = []
    shuffled = list(chunks)
    rng.shuffle(shuffled)

    for chunk in shuffled:
        text = chunk["text"].strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for s in sentences[:3]:
            s = s.strip()
            if len(s) < 30 or len(s) > 150:
                continue
            m = re.match(r"^([A-Z][A-Za-z\s\-]{3,40})\s+(?:is|are|es|son)\s+", s)
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
    return pairs


def generate_domain_pairs(
    chunks: list[dict],
    target: int = 2000,
    min_score: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Generate and filter training pairs from domain chunks."""
    # Generate candidates
    first_sent = _generate_first_sentence_pairs(chunks, max_pairs=target * 2, seed=seed)
    definitions = _generate_definition_pairs(chunks, max_pairs=target, seed=seed)
    questions = _generate_question_pairs(chunks, max_pairs=target, seed=seed)

    all_candidates = first_sent + definitions + questions
    logger.info(
        "Generated %d candidate pairs (first_sent=%d, def=%d, question=%d)",
        len(all_candidates),
        len(first_sent),
        len(definitions),
        len(questions),
    )

    if not all_candidates:
        return []

    # Cross-encoder filtering
    filtered = _filter_with_reranker(all_candidates, min_score=min_score)

    # Deduplicate
    seen = set()
    unique = []
    for p in filtered:
        h = hashlib.md5((p["query"][:200] + p["positive"][:200]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)

    random.Random(seed).shuffle(unique)
    return unique[:target]


def _filter_with_reranker(
    pairs: list[dict],
    min_score: float = 0.2,
    batch_size: int = 64,
) -> list[dict]:
    """Score pairs with cross-encoder and filter low quality."""
    from suyven_rag.rag.model_registry import get_reranker

    reranker = get_reranker()
    logger.info("Filtering %d pairs with reranker (min_score=%.2f)...", len(pairs), min_score)

    scored = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        inputs = [(p["query"], p["positive"]) for p in batch]
        scores = reranker.predict(inputs, show_progress_bar=False)
        for pair, score in zip(batch, scores, strict=False):
            pair["reranker_score"] = float(score)
            scored.append(pair)

    kept = [p for p in scored if p["reranker_score"] >= min_score]
    kept.sort(key=lambda x: x["reranker_score"], reverse=True)

    logger.info(
        "Reranker filter: %d/%d kept (%.1f%%)",
        len(kept),
        len(scored),
        100 * len(kept) / max(len(scored), 1),
    )
    return kept


# ---------------------------------------------------------------------------
# Step 3: Train LoRA on domain data
# ---------------------------------------------------------------------------


def train_domain_model(
    slug: str,
    pairs_path: Path,
    config: DomainFinetuneConfig,
) -> dict:
    """Train LoRA adapters on domain-specific pairs."""
    from suyven_rag.finetune.config import TrainConfig
    from suyven_rag.finetune.train import train

    output_dir = DOMAIN_FT_DIR / slug / "checkpoints"

    train_config = TrainConfig(
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        temperature=config.temperature,
        train_data_path=pairs_path,
        eval_split=config.eval_split,
        max_seq_length=config.max_seq_length,
        output_dir=output_dir,
        loss_plot_path=DOMAIN_FT_DIR / slug / "loss_curve.png",
    )

    logger.info(
        "[%s] Starting LoRA training (rank=%d, epochs=%d, lr=%s)",
        slug,
        config.lora_rank,
        config.epochs,
        config.learning_rate,
    )

    summary = train(train_config)
    return summary


# ---------------------------------------------------------------------------
# Step 4: Register domain model
# ---------------------------------------------------------------------------


def register_domain_model(slug: str) -> Path:
    """Register the fine-tuned model in model_registry for this domain.

    Returns the merged model path.
    """
    from suyven_rag.rag.model_registry import register_embed_model

    merged_path = DOMAIN_FT_DIR / slug / "checkpoints" / "merged_model"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged model not found: {merged_path}")

    model_name = f"domain_{slug}_embed"
    register_embed_model(model_name, str(merged_path))
    logger.info("[%s] Registered domain embed model: %s", slug, model_name)

    return merged_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_domain_finetune(
    slug: str,
    config: DomainFinetuneConfig | None = None,
) -> DomainFinetuneResult:
    """Run the full domain fine-tuning pipeline.

    1. Sample chunks from domain collection
    2. Generate self-supervised training pairs
    3. Filter with cross-encoder
    4. Train LoRA
    5. Merge and register model

    Returns DomainFinetuneResult with status and metrics.
    """
    if config is None:
        config = DomainFinetuneConfig(slug=slug)

    result = DomainFinetuneResult(slug=slug, status="running")

    try:
        # Verify domain exists
        from suyven_rag.rag.domain_registry import get_domain

        domain = get_domain(slug)
        logger.info("[%s] Starting fine-tune pipeline for domain: %s", slug, domain.name)

        # Step 1: Sample chunks
        chunks = sample_domain_chunks(slug)
        if len(chunks) < config.min_pairs:
            result.status = "insufficient_data"
            result.error = f"Domain has {len(chunks)} chunks, need at least {config.min_pairs}"
            logger.warning("[%s] %s", slug, result.error)
            return result

        # Step 2-3: Generate and filter pairs
        pairs = generate_domain_pairs(
            chunks,
            target=config.target_pairs,
            min_score=config.min_reranker_score,
        )
        result.pairs_generated = len(pairs)

        if len(pairs) < config.min_pairs:
            result.status = "insufficient_data"
            result.error = (
                f"Only {len(pairs)} quality pairs generated, need at least {config.min_pairs}"
            )
            logger.warning("[%s] %s", slug, result.error)
            return result

        result.pairs_after_filter = len(pairs)

        # Save pairs
        pairs_dir = DOMAIN_FT_DIR / slug
        pairs_dir.mkdir(parents=True, exist_ok=True)
        pairs_path = pairs_dir / "pairs.jsonl"

        with open(pairs_path, "w", encoding="utf-8") as f:
            for p in pairs:
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
        logger.info("[%s] Saved %d training pairs to %s", slug, len(pairs), pairs_path)

        # Step 4: Train
        t0 = time.time()
        summary = train_domain_model(slug, pairs_path, config)
        result.training_time_s = round(time.time() - t0, 1)
        result.train_pairs = summary.get("train_pairs", 0)
        result.eval_pairs = summary.get("eval_pairs", 0)
        result.final_train_loss = summary.get("final_train_loss")
        result.final_eval_loss = summary.get("final_eval_loss")
        result.lora_path = str(DOMAIN_FT_DIR / slug / "checkpoints" / "lora_weights.pt")

        # Step 5: Register model
        merged_path = register_domain_model(slug)
        result.model_path = str(merged_path)
        result.status = "success"

        # Update domain config with model path
        from suyven_rag.rag.domain_registry import update_domain

        update_domain(slug, chunk_count=len(chunks))

        logger.info(
            "[%s] Fine-tune complete: %d pairs, train_loss=%.4f, eval_loss=%.4f, time=%.1fs",
            slug,
            len(pairs),
            result.final_train_loss or 0,
            result.final_eval_loss or 0,
            result.training_time_s,
        )

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        logger.error("[%s] Fine-tune failed: %s", slug, e, exc_info=True)

    # Save result
    result_path = DOMAIN_FT_DIR / slug / "finetune_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Domain-specific embedding fine-tuning")
    parser.add_argument("slug", help="Domain slug (e.g., 'oncologia')")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--target-pairs", type=int, default=2000)
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=0.2)
    args = parser.parse_args()

    config = DomainFinetuneConfig(
        slug=args.slug,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.rank,
        target_pairs=args.target_pairs,
        min_pairs=args.min_pairs,
        min_reranker_score=args.min_score,
    )

    result = run_domain_finetune(args.slug, config)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
