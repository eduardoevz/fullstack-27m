"""Ejecuta una suite de 20 prompts del dominio y mide sintaxis y «en tema», con IC por bootstrap.

    python src/eval/run_suite.py --set test --samples 5 --json benchmarks/x.json
    python src/eval/run_suite.py --set dev --samples 1 --rep-penalty 1.1       # ajuste de decodificacion

`--set test` es el conjunto congelado (PROMPTS): no se usa para ajustar nada. `--set dev` (DEV_PROMPTS) es para ajustar.
Por prompt se generan `--samples` continuaciones; se mide si prompt+continuacion parsea (estricta y recortada, como en
syntax_check.py) y si la continuacion menciona algo esperado (heuristica «en tema», no mide correccion). El IC95 % es un
bootstrap sobre prompts. Con `--md` se escribe un informe legible con la primera muestra de cada prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.eval.harness import run_prompt_set, summarize_prompts
from src.eval.prompts import DEV_PROMPTS, PROMPTS
from src.eval.syntax_check import add_decoding_args, decoding_from_args
from src.sample import load_model


def fmt(ci: dict) -> str:
    return f"{ci['mean']:.0%} (IC95 {ci['lo']:.0%}-{ci['hi']:.0%})"


def write_md(path: str, results: list[dict], summary: dict, header: str) -> None:
    lines = ["# Suite de 20 prompts del dominio\n", header + "\n",
             f"**Resumen:** sintaxis estricta {fmt(summary['strict'])} · recortada {fmt(summary['trimmed'])} · "
             f"en tema {fmt(summary['on_topic'])} (prompts={summary['n_prompts']}, muestras={summary['n_samples']}).\n"]
    for r in (x for x in results if x["sample"] == 0):
        lines += [f"## {r['id']} — {r['description']}\n",
                  f"estricta: {'sí' if r['strict_ok'] else 'no'} · recortada: {'sí' if r['trimmed_ok'] else 'no'} · "
                  f"en tema: {'sí' if r['on_topic'] else 'no'}\n",
                  "```" + {"tsx": "tsx", "ts": "ts", "jsx": "jsx", "js": "js", "py": "python"}[r["lang"]],
                  r["prompt"] + "«« continuación del modelo »»" + r["completion"], "```\n"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--set", choices=["test", "dev"], default="test")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    add_decoding_args(ap)
    ap.set_defaults(temperature=0.7)                     # la suite de la Fase 7 se corrio con T=0,7
    args = ap.parse_args()
    decoding = decoding_from_args(args)

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    ckpt = args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt"
    model, step = load_model(ckpt, cfg)

    prompts = PROMPTS if args.set == "test" else DEV_PROMPTS
    results = run_prompt_set(model, tok, prompts, args.samples, args.max_new_tokens, decoding, args.seed)
    summary = summarize_prompts(results, seed=args.seed)
    if not args.quiet:
        for r in (x for x in results if x["sample"] == 0):
            print(f"{r['id']:<26} estricta={'si' if r['strict_ok'] else 'no':<3} recortada={'si' if r['trimmed_ok'] else 'no':<3} "
                  f"en_tema={'si' if r['on_topic'] else 'no'}")
    print(f"\n[{args.set}] estricta {fmt(summary['strict'])} | recortada {fmt(summary['trimmed'])} | "
          f"en tema {fmt(summary['on_topic'])}")

    header = (f"Modelo: `{Path(str(ckpt)).name}` (paso {step}). {decoding.label()}, hasta {args.max_new_tokens} tokens, "
              f"semilla base {args.seed}, {args.samples} muestra(s) por prompt. Sin selección: es lo primero que salió.")
    if args.md:
        write_md(args.md, results, summary, header)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "checkpoint": str(ckpt), "checkpoint_step": step, "set": args.set, "decoding": decoding.generate_kwargs(),
            "settings": {"samples": args.samples, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
            "summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("listo")


if __name__ == "__main__":
    main()
