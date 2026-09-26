"""Rotary Position Embedding (RoPE), convencion `rotate_half` de Llama.

RoPE rota q y k segun su posicion, de modo que el producto q.k depende solo de la
distancia relativa. No hay tabla de posiciones aprendida, asi que no existe un limite
duro de contexto: la tabla se extiende bajo demanda si T supera `max_len`.
"""

from __future__ import annotations

import torch
from torch import nn


def build_rope_cache(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos y sin de forma (max_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)   # (max_len, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)                                 # (max_len, head_dim)
    return angles.cos(), angles.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim) ya cortados a las posiciones de x."""
    return x * cos + _rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int, theta: float):
        super().__init__()
        self.head_dim, self.theta = head_dim, theta
        cos, sin = build_rope_cache(head_dim, max_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin de las posiciones [offset, offset + seq_len). El desfase servira a la cache KV."""
        end = offset + seq_len
        if end > self.cos.shape[0]:
            self.cos, self.sin = build_rope_cache(self.head_dim, max(end, 2 * self.cos.shape[0]), self.theta)
        return self.cos[offset:end], self.sin[offset:end]
