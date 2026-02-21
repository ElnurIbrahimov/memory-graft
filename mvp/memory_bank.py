"""
Persistent Memory Bank — a data structure, NOT a neural network.

Stores key-value pairs in the model's latent memory space.
Lives on disk. Survives restarts. Grows with use.
"""

import torch
import time
from pathlib import Path


class MemoryBank:
    def __init__(self, memory_dim=512, max_entries=256):
        self.memory_dim = memory_dim
        self.max_entries = max_entries
        self.keys = []           # List of [memory_dim] tensors (CPU)
        self.values = []         # List of [memory_dim] tensors (CPU)
        self.importance = []     # List of floats
        self.timestamps = []     # List of floats
        self.access_counts = []  # List of ints

    @property
    def size(self):
        return len(self.keys)

    def has_entries(self):
        return self.size > 0

    def clear(self):
        self.keys = []
        self.values = []
        self.importance = []
        self.timestamps = []
        self.access_counts = []

    def write(self, keys, values, importance_scores=None, detach=True):
        """
        Add memories.
        keys: [n, memory_dim] or [memory_dim]
        values: [n, memory_dim] or [memory_dim]
        importance_scores: [n] or scalar or None
        detach: if False, keep gradients (for Phase 2 training)
        """
        if keys.dim() == 1:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)

        n = keys.size(0)
        if importance_scores is None:
            importance_scores = [1.0] * n
        elif torch.is_tensor(importance_scores):
            if importance_scores.dim() == 0:
                importance_scores = [importance_scores.item()] * n
            else:
                importance_scores = importance_scores.tolist()

        now = time.time()
        for i in range(n):
            if detach:
                self.keys.append(keys[i].detach().cpu())
                self.values.append(values[i].detach().cpu())
            else:
                self.keys.append(keys[i])
                self.values.append(values[i])
            self.importance.append(importance_scores[i])
            self.timestamps.append(now)
            self.access_counts.append(0)

        if self.size > self.max_entries:
            self._evict()

    def get_keys(self, device="cpu"):
        """Return all keys as [n_memories, memory_dim] tensor."""
        if not self.has_entries():
            return None
        return torch.stack(self.keys).to(device)

    def get_values(self, device="cpu"):
        """Return all values as [n_memories, memory_dim] tensor."""
        if not self.has_entries():
            return None
        return torch.stack(self.values).to(device)

    def save(self, path):
        """Persist to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": self.keys,
                "values": self.values,
                "importance": self.importance,
                "timestamps": self.timestamps,
                "access_counts": self.access_counts,
                "memory_dim": self.memory_dim,
                "max_entries": self.max_entries,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        """Load from disk."""
        data = torch.load(path, weights_only=False)
        bank = cls(
            memory_dim=data["memory_dim"],
            max_entries=data["max_entries"],
        )
        bank.keys = data["keys"]
        bank.values = data["values"]
        bank.importance = data["importance"]
        bank.timestamps = data["timestamps"]
        bank.access_counts = data["access_counts"]
        return bank

    def _evict(self):
        """Remove lowest-scoring memories to stay under max_entries."""
        now = time.time()
        scores = []
        for i in range(self.size):
            age = max(now - self.timestamps[i], 1.0)
            score = self.importance[i] * (self.access_counts[i] + 1) / age
            scores.append(score)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.max_entries]
        top_indices.sort()  # maintain insertion order

        self.keys = [self.keys[i] for i in top_indices]
        self.values = [self.values[i] for i in top_indices]
        self.importance = [self.importance[i] for i in top_indices]
        self.timestamps = [self.timestamps[i] for i in top_indices]
        self.access_counts = [self.access_counts[i] for i in top_indices]

    def __repr__(self):
        return f"MemoryBank(entries={self.size}, dim={self.memory_dim}, max={self.max_entries})"
