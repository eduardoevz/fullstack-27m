"""Exporta solo los pesos de un checkpoint de entrenamiento (sin optimizador, sampler ni RNG) para publicarlos.

    python src/export_weights.py --ckpt C:/llm-fullstack-data/checkpoints_v2_long_c_hitos/hito-0018310.pt --out release/fullstack-27m-v2.pt

El archivo resultante se carga con torch.load(..., weights_only=True) y con src.sample.load_model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def export_weights(ckpt: str | Path, out: str | Path) -> dict:
    """Guarda {"model": state_dict, "step": paso}. Devuelve paso, nº de tensores y tamaño en bytes."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = dict(state["model"])      # sin clonar: lm_head y tok_emb comparten almacenamiento y se guardan una sola vez
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": weights, "step": state.get("step")}, out)
    return {"step": state.get("step"), "tensors": len(weights), "bytes": out.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    info = export_weights(args.ckpt, args.out)
    print(f"paso {info['step']} · {info['tensors']} tensores · {info['bytes'] / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
