"""pass@1 funcional: el modelo completa el cuerpo de 30 funciones pequenas y se ejecutan sus tests unitarios.

    python src/eval/run_tasks.py --samples 5 --json benchmarks/x.json
    python src/eval/run_tasks.py --samples 5 --rep-penalty 1.1

Ver src/eval/tasks.py para las tareas y para las medidas de seguridad (se ejecuta codigo generado por el modelo).
Por defecto T=0,2 (casi determinista), que es lo habitual para medir pass@1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.eval.harness import FRAMING  # noqa: F401  (misma convencion de lenguajes que la suite)
from src.eval.stats import bootstrap_ci, pass_at_k
from src.eval.syntax_check import add_decoding_args, decoding_from_args
from src.eval.tasks import TASKS, run_task, truncate_completion
from src.sample import generate, load_model


def summarize_tasks(rows: list[dict], seed: int = 0) -> dict:
    """pass@1 por tarea (n muestras, c aciertos) y su media con IC95 % por bootstrap sobre tareas; total y por lenguaje."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["id"], []).append(r)
    per_task = {tid: (pass_at_k(len(rs), sum(r["passed"] for r in rs), 1), rs[0]["lang"]) for tid, rs in by_task.items()}

    def block(items):
        mean, lo, hi = bootstrap_ci([p for p, _ in items], seed=seed)
        return {"mean": mean, "lo": lo, "hi": hi}

    out = {"n_tasks": len(by_task), "n_samples": len(rows), "pass_at_1": block(list(per_task.values())),
           "tasks_solved_at_least_once": sum(any(r["passed"] for r in rs) for rs in by_task.values())}
    for lang in sorted({l for _, l in per_task.values()}):
        out[lang] = {"n_tasks": sum(1 for _, l in per_task.values() if l == lang),
                     "pass_at_1": block([it for it in per_task.values() if it[1] == lang])}
    return out


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    add_decoding_args(ap)
    ap.set_defaults(temperature=0.2)
    args = ap.parse_args()
    decoding = decoding_from_args(args)

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    ckpt = args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt"
    model, step = load_model(ckpt, cfg)
    eot = tok.token_to_id("<|endoftext|>")
    head = {"py": [tok.token_to_id("<|file|>"), tok.token_to_id("<|lang_py|>")],
            "js": [tok.token_to_id("<|file|>"), tok.token_to_id("<|lang_js|>")]}

    rows, t0 = [], time.time()
    for j in range(args.samples):
        for i, task in enumerate(TASKS):
            ids = head[task["lang"]] + tok.encode(task["prompt"]).ids
            g = torch.Generator().manual_seed(args.seed + 1000 * j + i)
            new = generate(model, ids, args.max_new_tokens, eos_id=eot, generator=g, **decoding.generate_kwargs())
            completion = tok.decode(new, skip_special_tokens=False)
            passed, reason = run_task(task, completion)
            rows.append({"id": task["id"], "lang": task["lang"], "sample": j, "passed": passed, "reason": reason[:200],
                         "completion": truncate_completion(completion, task["lang"])})
        print(f"\rmuestra {j + 1}/{args.samples}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    summary = summarize_tasks(rows, seed=args.seed)
    p = summary["pass_at_1"]
    print(f"pass@1 = {p['mean']:.1%} (IC95 {p['lo']:.1%}-{p['hi']:.1%}) | py {summary['py']['pass_at_1']['mean']:.1%} | "
          f"js {summary['js']['pass_at_1']['mean']:.1%} | tareas resueltas al menos una vez: "
          f"{summary['tasks_solved_at_least_once']}/{summary['n_tasks']}")
    unsafe = sum(1 for r in rows if r["reason"].startswith("unsafe"))
    print(f"rechazadas por el filtro de seguridad: {unsafe} de {len(rows)}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"checkpoint": str(ckpt), "checkpoint_step": step,
            "decoding": decoding.generate_kwargs(), "settings": {"samples": args.samples,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed}, "summary": summary,
            "rejected_by_safety_filter": unsafe, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"guardado en {args.json}")


if __name__ == "__main__":
    main()
