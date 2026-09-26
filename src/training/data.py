"""Muestreo de batches por offsets aleatorios sobre un memmap uint16 (sin DataLoader ni workers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BatchSampler:
    def __init__(self, bin_path: str | Path, block_size: int, batch_size: int, seed: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size, self.batch_size = block_size, batch_size
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{bin_path} tiene {len(self.data)} tokens; se necesitan mas de {block_size + 1}")
        self.rng = np.random.default_rng(seed)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        offsets = self.rng.integers(0, len(self.data) - self.block_size - 1, size=self.batch_size)
        rows = np.stack([self.data[o:o + self.block_size + 1] for o in offsets]).astype(np.int64)
        t = torch.from_numpy(rows)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()

    def state_dict(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]
