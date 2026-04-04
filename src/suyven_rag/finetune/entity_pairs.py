"""Entity-aware training pair generation (inspired by GraphRAG/MiroFish).

Extracts entities and relationships from chunks using spaCy NER,
then generates training pairs where:
  - Query = "What is [ENTITY]?" or "How does [ENTITY] relate to [ENTITY2]?"
  - Positive = chunk that defines/explains the entity

This produces much higher quality pairs than heuristic approaches because
entities represent real concepts users would search for.

Usage:
    python -m finetune.entity_pairs
    python -m finetune.entity_pairs --max-pairs 2000 --min-score 0.3
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "data" / "finetune" / "entity_pairs.jsonl"

# Technical entity patterns (no spaCy needed — regex is faster and domain-specific)
ENTITY_PATTERNS = [
    # AWS services
    r"\b(Amazon\s+\w+|AWS\s+\w+|S3|EC2|Lambda|DynamoDB|SageMaker|CloudFormation|ECS|EKS|RDS|SNS|SQS|IAM|VPC|CloudWatch|Kinesis|Redshift|Glue|Athena|EMR|Step\s+Functions)\b",
    # ML/AI terms
    r"\b(transformer|attention\s+mechanism|BERT|GPT|LLM|embedding|fine-tuning|LoRA|RAG|retrieval|reranker|cross-encoder|bi-encoder|tokenizer|softmax|gradient\s+descent|backpropagation|neural\s+network|CNN|RNN|LSTM|GAN|VAE|diffusion|reinforcement\s+learning)\b",
    # Infrastructure
    r"\b(Docker|Kubernetes|Terraform|Ansible|Jenkins|GitHub\s+Actions|CI/CD|microservices|load\s+balancer|auto\s*scaling|serverless|container|pod|node|cluster|ingress|service\s+mesh)\b",
    # Data
    r"\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|Apache\s+Kafka|Apache\s+Spark|Hadoop|ChromaDB|Pinecone|Weaviate|FAISS|vector\s+database|data\s+lake|data\s+warehouse|ETL|CDC)\b",
    # Python/Programming
    r"\b(FastAPI|Flask|Django|React|Vue\.js|PyTorch|TensorFlow|pandas|numpy|scikit-learn|sentence-transformers|HuggingFace|Transformers)\b",
    # Concepts
    r"\b(CAP\s+theorem|ACID|BASE|MapReduce|sharding|partitioning|replication|consensus|Raft|Paxos|eventual\s+consistency|strong\s+consistency|idempotency|circuit\s+breaker|rate\s+limiting|caching|CDN)\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ENTITY_PATTERNS]


def extract_entities(text: str) -> list[str]:
    """Extract technical entities from text using domain-specific patterns."""
    entities = set()
    for pattern in COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(0).strip()
            if len(entity) >= 2:
                entities.add(entity)
    return list(entities)


def load_corpus() -> list[dict]:
    """Load all chunks from ChromaDB."""
    from suyven_rag.rag.index_registry import get_index

    col = get_index()
    total = col.count()
    logger.info("Loading %d chunks...", total)

    all_docs = []
    batch_size = 5000
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
    return all_docs


def generate_entity_query_pairs(
    chunks: list[dict],
    max_pairs: int = 5000,
) -> list[dict]:
    """Generate (query, passage) pairs from entity extraction.

    For each chunk, extract entities and generate queries like:
    - "What is [ENTITY]?"
    - "How does [ENTITY] work?"
    - "Explain [ENTITY]"
    """
    entity_to_chunks: dict[str, list[dict]] = defaultdict(list)

    for chunk in chunks:
        entities = extract_entities(chunk["text"])
        for entity in entities:
            entity_to_chunks[entity.lower()].append(chunk)

    logger.info("Found %d unique entities across %d chunks", len(entity_to_chunks), len(chunks))

    # Generate pairs
    pairs = []
    query_templates = [
        "What is {}?",
        "How does {} work?",
        "Explain {}",
        "{} overview",
        "{} best practices",
    ]

    for entity, entity_chunks in sorted(entity_to_chunks.items(), key=lambda x: -len(x[1])):
        if len(entity) < 3:
            continue
        # Use the chunk that mentions the entity earliest (likely the definition)
        for chunk in entity_chunks[:3]:
            for template in query_templates[:2]:
                query = template.format(entity)
                pairs.append(
                    {
                        "query": query,
                        "positive": chunk["text"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "strategy": "entity_query",
                        "entity": entity,
                    }
                )
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    logger.info("Generated %d entity-query pairs", len(pairs))
    return pairs


def generate_entity_relationship_pairs(
    chunks: list[dict],
    max_pairs: int = 2000,
) -> list[dict]:
    """Generate pairs from co-occurring entities (relationship queries).

    If two entities appear in the same chunk, generate:
    - "What is the relationship between [A] and [B]?"
    - "How does [A] relate to [B]?"
    """
    pairs = []

    for chunk in chunks:
        entities = extract_entities(chunk["text"])
        if len(entities) < 2:
            continue

        # Generate relationship queries for entity pairs
        for i in range(min(len(entities), 4)):
            for j in range(i + 1, min(len(entities), 4)):
                e1, e2 = entities[i], entities[j]
                if e1.lower() == e2.lower():
                    continue
                query = f"How does {e1} relate to {e2}?"
                pairs.append(
                    {
                        "query": query,
                        "positive": chunk["text"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "strategy": "entity_relationship",
                        "entity": f"{e1} + {e2}",
                    }
                )
                if len(pairs) >= max_pairs:
                    return pairs

    logger.info("Generated %d entity-relationship pairs", len(pairs))
    return pairs


def filter_with_reranker(
    pairs: list[dict],
    min_score: float = 0.3,
    batch_size: int = 64,
) -> list[dict]:
    """Score and filter pairs with cross-encoder reranker."""
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

    avg_all = np.mean([p["reranker_score"] for p in scored]) if scored else 0
    avg_kept = np.mean([p["reranker_score"] for p in kept]) if kept else 0
    logger.info(
        "Reranker: %d/%d kept (%.1f%%), avg=%.3f, avg_kept=%.3f",
        len(kept),
        len(scored),
        100 * len(kept) / max(len(scored), 1),
        avg_all,
        avg_kept,
    )
    return kept


def run(max_pairs: int = 3000, min_score: float = 0.3, output: Path = OUTPUT):
    """Generate entity-aware training pairs."""
    chunks = load_corpus()

    entity_queries = generate_entity_query_pairs(chunks, max_pairs=max_pairs * 2)
    relationship_queries = generate_entity_relationship_pairs(chunks, max_pairs=max_pairs)

    all_candidates = entity_queries + relationship_queries
    logger.info("Total entity candidates: %d", len(all_candidates))

    filtered = filter_with_reranker(all_candidates, min_score=min_score)

    # Deduplicate
    import hashlib

    seen = set()
    unique = []
    for p in filtered:
        h = hashlib.md5((p["query"] + p["positive"][:200]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)

    unique = unique[:max_pairs]

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

    logger.info("Saved %d entity pairs to %s", len(unique), output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(max_pairs=args.max_pairs, min_score=args.min_score, output=args.output)


if __name__ == "__main__":
    main()
