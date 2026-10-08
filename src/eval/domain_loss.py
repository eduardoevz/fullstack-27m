"""Perdida de un checkpoint sobre archivos de tokens (val.bin, val_<dominio>.bin). Fase 9, paso 7.

Ventanas consecutivas de `block` tokens (+1 de objetivo), sin solaparse; si hay mas de
`max_windows` se elige un subconjunto con semilla fija. Es la linea base por dominio de la Fase 11.

    python -m src.eval.domain_loss --tokens-dir C:/llm-fullstack-data/tokens_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def stream_windows(data: np.ndarray, block: int, max_windows: int | None = None, seed: int = 0) -> np.ndarray:
    n = (len(data) - 1) // block
    n = max(0, min(n, (len(data) - block - 1) // block + 1))
    starts = np.arange(n) * block
    if max_windows is not None and n > max_windows:
        starts = np.sort(np.random.default_rng(seed).choice(starts, size=max_windows, replace=False))
    return np.stack([data[s:s + block + 1] for s in starts]).astype(np.int64) if len(starts) else np.empty((0, block + 1), np.int64)


@torch.no_grad()
def stream_loss(model, data: np.ndarray, block: int = 512, batch: int = 16, max_windows: int | None = None, seed: int = 0) -> dict:
    rows = stream_windows(data, block, max_windows, seed)
    total, count = 0.0, 0
    for i in range(0, len(rows), batch):
        t = torch.from_numpy(rows[i:i + batch])
        logits, _ = model(t[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction="sum")
        total += float(loss)
        count += t[:, 1:].numel()
    return {"loss": total / max(1, count), "windows": int(len(rows)), "tokens": count}


def main() -> None:
    from src.config import load_config
    from src.sample import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens-dir", required=True)
    ap.add_argument("--ckpt", default="v1-best.pt")
    ap.add_argument("--max-windows", type=int, default=1500)
    ap.add_argument("--out", default="benchmarks/v1_domain_loss.json")
    args = ap.parse_args()

    cfg = load_config()
    model, step = load_model(Path(cfg.paths.checkpoint_dir) / args.ckpt, cfg)
    block = cfg.model.block_size
    out = {"checkpoint": args.ckpt, "step": step, "block": block, "max_windows": args.max_windows, "sets": {}}
    files = {"val_v2 (mezcla v2)": Path(args.tokens_dir) / "val.bin",
             "val_v1 (mezcla v1)": Path(cfg.paths.token_dir) / "val.bin"}
    for p in sorted(Path(args.tokens_dir).glob("val_*.bin")):
        files[p.stem.replace("val_", "dominio:")] = p
    for name, p in files.items():
        data = np.fromfile(p, dtype=np.uint16)
        r = stream_loss(model, data, block=block, max_windows=args.max_windows)
        out["sets"][name] = {**r, "file_tokens": int(len(data))}
        print(f"{name:24s} loss={r['loss']:.4f}  ({r['windows']} ventanas, {r['tokens']:,} tokens)", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
