"""Un paso de optimizacion con acumulacion de gradiente, y evaluacion en validacion."""

from __future__ import annotations

import torch

from src.config import TrainingConfig
from src.training.data import BatchSampler
from src.training.schedule import get_lr


def accumulate_gradients(model, batches) -> float:
    """Suma gradientes de varios micro-batches (cada perdida dividida por su numero). Devuelve la perdida media."""
    batches = list(batches)
    total = 0.0
    for x, y in batches:
        _, loss = model(x, targets=y)
        (loss / len(batches)).backward()
        total += loss.item()
    return total / len(batches)


def run_step(model, optimizer, sampler: BatchSampler, cfg: TrainingConfig, step: int) -> tuple[float, float]:
    """Un paso completo. Devuelve (perdida media, norma del gradiente antes del recorte)."""
    model.train()
    for g in optimizer.param_groups:
        g["lr"] = get_lr(step, cfg)
    optimizer.zero_grad(set_to_none=True)
    loss = accumulate_gradients(model, (sampler.get_batch() for _ in range(cfg.grad_accum_steps)))
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
    optimizer.step()
    return loss, gnorm


@torch.no_grad()
def evaluate(model, sampler: BatchSampler, n_batches: int) -> float:
    model.eval()
    total = sum(model(*sampler.get_batch())[1].item() for _ in range(n_batches))
    model.train()
    return total / n_batches
