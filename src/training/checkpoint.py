"""Checkpoints con escritura atomica y rotacion (los N ultimos + el mejor por validacion)."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def save_checkpoint(path: str | Path, model, optimizer, sampler, step: int, best_val: float) -> None:
    """Guarda el estado completo. Escribe a un .tmp y lo renombra: un corte no deja un checkpoint roto."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, model, optimizer, sampler) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    return {"step": state["step"], "best_val": state["best_val"]}


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    files = sorted(Path(ckpt_dir).glob("ckpt-*.pt"))
    return files[-1] if files else None


def rotate_checkpoints(ckpt_dir: str | Path, keep: int) -> None:
    """Borra los ckpt-*.pt salvo los `keep` mas recientes. `best.pt` no se toca."""
    files = sorted(Path(ckpt_dir).glob("ckpt-*.pt"))
    for old in files[:-keep] if keep > 0 else files:
        old.unlink()
