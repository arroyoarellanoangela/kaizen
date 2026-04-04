"""Intrinsic eval: does the fine-tuned model better distinguish correct vs wrong passages?"""

import json
import random

import numpy as np
from sentence_transformers import SentenceTransformer

PAIRS_PATH = "data/finetune/pairs_v2.jsonl"
CHECKPOINT = "data/finetune/checkpoints/merged_model"


def cos_sim(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    pairs = []
    with open(PAIRS_PATH, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    random.seed(42)
    random.shuffle(pairs)
    eval_pairs = pairs[-296:]

    queries = [p["query"] for p in eval_pairs]
    positives = [p["positive"] for p in eval_pairs]
    negatives = positives[1:] + positives[:1]

    base = SentenceTransformer("BAAI/bge-m3").cuda().half()
    ft = SentenceTransformer(CHECKPOINT).cuda().half()

    print("Encoding with both models...")
    bq = base.encode(queries, show_progress_bar=False, convert_to_numpy=True)
    bp = base.encode(positives, show_progress_bar=False, convert_to_numpy=True)
    bn = base.encode(negatives, show_progress_bar=False, convert_to_numpy=True)
    fq = ft.encode(queries, show_progress_bar=False, convert_to_numpy=True)
    fp = ft.encode(positives, show_progress_bar=False, convert_to_numpy=True)
    fn = ft.encode(negatives, show_progress_bar=False, convert_to_numpy=True)

    base_pos = cos_sim(bq, bp)
    base_neg = cos_sim(bq, bn)
    ft_pos = cos_sim(fq, fp)
    ft_neg = cos_sim(fq, fn)

    base_margin = base_pos - base_neg
    ft_margin = ft_pos - ft_neg

    base_acc = np.mean(base_pos > base_neg) * 100
    ft_acc = np.mean(ft_pos > ft_neg) * 100

    print()
    print("=== INTRINSIC EVAL (296 pairs, correct vs wrong passage) ===")
    print()
    print("Metric                         Base    FineTuned    Delta")
    print("-" * 60)
    print(
        f"Avg cos(q, positive)         {np.mean(base_pos):.4f}     {np.mean(ft_pos):.4f}   {np.mean(ft_pos) - np.mean(base_pos):+.4f}"
    )
    print(
        f"Avg cos(q, negative)         {np.mean(base_neg):.4f}     {np.mean(ft_neg):.4f}   {np.mean(ft_neg) - np.mean(base_neg):+.4f}"
    )
    print(
        f"Avg margin (pos - neg)       {np.mean(base_margin):.4f}     {np.mean(ft_margin):.4f}   {np.mean(ft_margin) - np.mean(base_margin):+.4f}"
    )
    print(
        f"Accuracy (pos > neg)         {base_acc:.1f}%     {ft_acc:.1f}%   {ft_acc - base_acc:+.1f}%"
    )

    m = np.mean(base_margin)
    if m > 0.001:
        print(f"\nMargin improvement: {((np.mean(ft_margin) / m) - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()
