"""Tasa de aprendizaje: calentamiento lineal y despues coseno hasta el minimo."""

from __future__ import annotations

import math

from src.config import TrainingConfig


def get_lr(step: int, cfg: TrainingConfig) -> float:
    """LR del paso `step` (base 0). El paso `warmup_steps - 1` ya alcanza el pico."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_learning_rate
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)
