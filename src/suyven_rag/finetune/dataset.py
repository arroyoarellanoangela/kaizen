"""PyTorch Dataset for contrastive training pairs."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ContrastivePairsDataset(Dataset):
    """Dataset of (query, positive_passage) pairs for contrastive learning.

    Each item in the JSONL file has:
        {"query": "...", "positive": "...", "source": "...", "category": "..."}

    Returns (query_text, positive_text) tuples.
    """

    def __init__(self, path: Path, max_samples: int = 0):
        self.pairs: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.pairs.append((entry["query"], entry["positive"]))
                if max_samples and len(self.pairs) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


class TripletDataset(Dataset):
    """Dataset of (query, positive, negative) triplets for hard negative training.

    Each item in the JSONL file has:
        {"query": "...", "positive": "...", "negative": "...", ...}

    Returns (query_text, positive_text, negative_text) tuples.
    """

    def __init__(self, path: Path, max_samples: int = 0):
        self.triplets: list[tuple[str, str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                self.triplets.append((entry["query"], entry["positive"], entry["negative"]))
                if max_samples and len(self.triplets) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.triplets[idx]


def train_eval_split(
    dataset: ContrastivePairsDataset,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple["ContrastivePairsDataset", "ContrastivePairsDataset"]:
    """Split dataset into train and eval subsets."""
    n = len(dataset)
    n_eval = max(1, int(n * eval_ratio))
    n_train = n - n_eval

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_ds = ContrastivePairsDataset.__new__(ContrastivePairsDataset)
    train_ds.pairs = [dataset.pairs[i] for i in indices[:n_train]]

    eval_ds = ContrastivePairsDataset.__new__(ContrastivePairsDataset)
    eval_ds.pairs = [dataset.pairs[i] for i in indices[n_train:]]

    return train_ds, eval_ds
