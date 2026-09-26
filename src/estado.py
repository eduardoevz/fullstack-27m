"""Resumen del entrenamiento: paso, perdida, velocidad y tiempo restante, leidos de logs/train.csv.

    python src/estado.py
"""

from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, load_config


def summarize(log_path: str | Path, max_steps: int, tokens_per_step: int) -> dict | None:
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    train, val = {}, {}
    with open(log_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["loss"]:
                train[int(r["step"])] = r             # tras reanudar, un paso puede repetirse: manda el ultimo
            if r["val_loss"]:
                val[int(r["step"])] = float(r["val_loss"])
    if not train:
        return None
    step = max(train)
    last = train[step]
    recent = [float(train[s]["tok_s"]) for s in sorted(train)[-5:]]
    tok_s = statistics.mean(recent)
    remaining = max(0, max_steps - step)
    return {
        "step": step, "max_steps": max_steps, "loss": float(last["loss"]), "tok_s": tok_s,
        "val_loss": val[max(val)] if val else None, "best_val": min(val.values()) if val else None,
        "remaining_steps": remaining, "days_left": remaining * tokens_per_step / tok_s / 86400,
        "log_age_min": (time.time() - log_path.stat().st_mtime) / 60,
    }


def main() -> None:
    cfg = load_config()
    s = summarize(REPO_ROOT / "logs" / "train.csv", cfg.training.max_steps,
                  cfg.training.tokens_per_step(cfg.model.block_size))
    if s is None:
        print("Todavia no hay registro de entrenamiento (logs/train.csv).")
        return
    print(f"Paso {s['step']:,} de {s['max_steps']:,}  ({100 * s['step'] / s['max_steps']:.1f} %)")
    print(f"Perdida de entrenamiento: {s['loss']:.4f}")
    if s["val_loss"] is not None:
        print(f"Perdida de validacion:    {s['val_loss']:.4f}  (mejor {s['best_val']:.4f})")
    print(f"Velocidad reciente:       {s['tok_s']:.0f} tok/s")
    print(f"Faltan {s['remaining_steps']:,} pasos: ~{s['days_left']:.1f} dias de maquina encendida a este ritmo")
    print(f"Ultimo registro hace {s['log_age_min']:.0f} min"
          + ("  -> parece detenido" if s["log_age_min"] > 15 else ""))


if __name__ == "__main__":
    main()
