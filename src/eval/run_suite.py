"""Ejecuta la suite de 20 prompts del dominio y guarda las continuaciones.

    python src/eval/run_suite.py

Por prompt: genera (T=0.7), mide si prompt+continuacion parsea (estricta y recortada, como en
syntax_check.py) y si la continuacion menciona algo esperado (heuristica «en tema»).
Escribe  samples/suite-20-prompts.md  (legible) y  benchmarks/prompt_suite.json  (datos).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.eval.prompts import PROMPTS
from src.eval.syntax import RETENTION_MIN, check_fragments, trim_to_valid
from src.sample import frame_prompt, generate, load_model

FRAMING = {"js": "js", "jsx": "js", "ts": "ts", "tsx": "ts", "py": "py"}


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--md", default=str(REPO_ROOT / "samples" / "suite-20-prompts.md"))
    ap.add_argument("--json", default=str(REPO_ROOT / "benchmarks" / "prompt_suite.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    eot = tok.token_to_id("<|endoftext|>")

    results = []
    for i, p in enumerate(PROMPTS):
        ids = frame_prompt(tok, p["prompt"], FRAMING[p["lang"]])
        new = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p, eot,
                       torch.Generator().manual_seed(args.seed + i))
        completion = tok.decode(new, skip_special_tokens=False)
        full = p["prompt"] + completion
        ok = check_fragments([{"id": 0, "lang": p["lang"], "code": full}])[0]["ok"]
        retention = 1.0 if ok else trim_to_valid(full, p["lang"])[1]
        results.append({"id": p["id"], "lang": p["lang"], "description": p["description"], "prompt": p["prompt"],
                        "completion": completion, "finished": len(new) < args.max_new_tokens,
                        "strict_ok": ok, "trimmed_ok": ok or retention >= RETENTION_MIN, "retention": retention,
                        "on_topic": any(e in completion for e in p["expect"])})
        print(f"{p['id']:<24} estricta={'si' if ok else 'no':<3} recortada={'si' if results[-1]['trimmed_ok'] else 'no':<3} "
              f"en_tema={'si' if results[-1]['on_topic'] else 'no'}")

    n = len(results)
    summary = {"n": n, "strict": sum(r["strict_ok"] for r in results) / n,
               "trimmed": sum(r["trimmed_ok"] for r in results) / n,
               "on_topic": sum(r["on_topic"] for r in results) / n}
    print(f"\nestricta {summary['strict']:.0%} | recortada {summary['trimmed']:.0%} | en tema {summary['on_topic']:.0%}")

    lines = [f"# Suite de 20 prompts del dominio\n",
             f"Modelo: `best.pt` (paso {step}). T={args.temperature}, top-k {args.top_k}, top-p {args.top_p}, "
             f"hasta {args.max_new_tokens} tokens, semilla base {args.seed}. Sin selección: es lo primero que salió.\n",
             f"**Resumen:** sintaxis estricta {summary['strict']:.0%} · recortada {summary['trimmed']:.0%} · "
             f"en tema {summary['on_topic']:.0%} (n={n}).\n"]
    for r in results:
        lines.append(f"## {r['id']} — {r['description']}\n")
        lines.append(f"estricta: {'sí' if r['strict_ok'] else 'no'} · recortada: {'sí' if r['trimmed_ok'] else 'no'} · "
                     f"en tema: {'sí' if r['on_topic'] else 'no'}\n")
        lines.append("```" + {"tsx": "tsx", "ts": "ts", "jsx": "jsx", "js": "js", "py": "python"}[r["lang"]])
        lines.append(r["prompt"] + "«« continuación del modelo »»" + r["completion"])
        lines.append("```\n")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps({"checkpoint_step": step, "settings": {k: getattr(args, k) for k in
        ("max_new_tokens", "temperature", "top_k", "top_p", "seed")}, "summary": summary, "results": results},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Guardado: {args.md} y {args.json}")


if __name__ == "__main__":
    main()
