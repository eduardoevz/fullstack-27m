"""AdamW con weight decay solo en matrices (no en normas ni en el embedding atado)."""

from __future__ import annotations

import torch

from src.config import TrainingConfig


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig) -> torch.optim.AdamW:
    emb_id = id(model.tok_emb.weight)
    decay, no_decay = [], []
    for p in model.parameters():                # parameters() no repite el peso atado
        (decay if p.ndim >= 2 and id(p) != emb_id else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))
