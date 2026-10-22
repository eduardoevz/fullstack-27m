"""Fase 13: mide si la cuantizacion dinamica int8 de las capas lineales acelera la inferencia en esta CPU.

    python src/eval/quant_bench.py --config config/model_v2_long_c.json --ckpt <hito-0018310.pt> --out benchmarks/v2_quant_bench.json

Puerta fijada ANTES de medir: se entrega solo si es >= 1,3x mas rapida que fp32. Condiciones de la Fase 7: prompt de 200 tokens,
100 tokens nuevos con cache KV, 4 hilos; mediana de `--repeats` repeticiones. Se mide tambien la variante que deja `lm_head`
(pesos atados al embedding) en fp32. La degradacion de calidad solo se mide si la puerta de velocidad se cumple.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.sample import generate, load_model

MIN_SPEEDUP = 1.3
PROMPT_TOKENS, NEW_TOKENS = 200, 100


def quantize(model: nn.Module, keep_lm_head: bool) -> nn.Module:
    q = copy.deepcopy(model)
    if keep_lm_head:
        head, q.lm_head = q.lm_head, nn.Identity()          # fuera de la cuantizacion; se repone despues
    q = torch.ao.quantization.quantize_dynamic(q, {nn.Linear}, dtype=torch.qint8)
    if keep_lm_head:
        q.lm_head = head
    return q


def tokens_per_second(model: nn.Module, repeats: int) -> float:
    prompt = list(range(5, 5 + PROMPT_TOKENS))
    rates = []
    for _ in range(repeats):
        t0 = time.time()
        out = generate(model, prompt, NEW_TOKENS, temperature=0.0)
        rates.append(len(out) / (time.time() - t0))
    return sorted(rates)[len(rates) // 2]


def size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "v2_quant_bench.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    rows = {"fp32": {"tok_s": tokens_per_second(model, args.repeats), "mb": size_mb(model)}}
    for name, keep in (("int8", False), ("int8_lm_head_fp32", True)):
        q = quantize(model, keep)
        rows[name] = {"tok_s": tokens_per_second(q, args.repeats), "mb": size_mb(q)}
    base = rows["fp32"]["tok_s"]
    for r in rows.values():
        r["speedup"] = r["tok_s"] / base
    passed = any(rows[k]["speedup"] >= MIN_SPEEDUP for k in ("int8", "int8_lm_head_fp32"))
    result = {"checkpoint_step": step, "threads": cfg.training.num_threads, "engine": torch.backends.quantized.engine,
              "torch": torch.__version__, "prompt_tokens": PROMPT_TOKENS, "new_tokens": NEW_TOKENS, "repeats": args.repeats,
              "min_speedup": MIN_SPEEDUP, "rows": rows, "gate_passed": passed}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, r in rows.items():
        print(f"{name:<18} {r['tok_s']:6.1f} tok/s  x{r['speedup']:.2f}  {r['mb']:.1f} MB")
    print("puerta (>= %.1fx): %s" % (MIN_SPEEDUP, "CUMPLE" if passed else "NO se cumple"))


if __name__ == "__main__":
    main()
