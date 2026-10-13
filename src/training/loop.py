"""Un paso de optimizacion con acumulacion de gradiente, y evaluacion en validacion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import TrainingConfig
from src.training.data import BatchSampler
from src.training.schedule import get_lr


def accumulate_gradients(model, batches, stop_weight: float = 1.0, spec=None) -> float:
    """Suma gradientes de varios micro-batches (cada perdida dividida por su numero). Devuelve la perdida media.

    Con `stop_weight` != 1 y un `FimSpec`, el <|endoftext|> que cierra el medio de las filas FIM pesa `stop_weight`
    veces en el gradiente (misma normalizacion por token). El valor devuelto es la perdida SIN ponderar."""
    batches = list(batches)
    total = 0.0
    for x, y in batches:
        logits, loss = model(x, targets=y)
        objective = loss
        if spec is not None and stop_weight != 1.0:
            from src.training.fim import middle_stop_mask
            mask = middle_stop_mask(x, y, spec).reshape(-1)
            if mask.any():
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[mask], y.reshape(-1)[mask], reduction="sum")
                objective = loss + (stop_weight - 1.0) * ce / mask.numel()
        (objective / len(batches)).backward()
        total += loss.item()
    return total / len(batches)


def run_step(model, optimizer, sampler: BatchSampler, cfg: TrainingConfig, step: int,
             spec=None) -> tuple[float, float]:
    """Un paso completo. Devuelve (perdida media, norma del gradiente antes del recorte)."""
    model.train()
    for g in optimizer.param_groups:
        g["lr"] = get_lr(step, cfg)
    optimizer.zero_grad(set_to_none=True)
    loss = accumulate_gradients(model, (sampler.get_batch() for _ in range(cfg.grad_accum_steps)),
                                cfg.fim_stop_weight, spec)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
    optimizer.step()
    return loss, gnorm


@torch.no_grad()
def evaluate(model, sampler: BatchSampler, n_batches: int) -> float:
    model.eval()
    total = sum(model(*sampler.get_batch())[1].item() for _ in range(n_batches))
    model.train()
    return total / n_batches
