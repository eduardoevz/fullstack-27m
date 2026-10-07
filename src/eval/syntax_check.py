"""Metrica de validez sintactica: genera N fragmentos desde cero y mide que porcentaje parsea.

    python src/eval/syntax_check.py --n 100
    python src/eval/syntax_check.py --n 100 --max-new-tokens 512 --rep-penalty 1.1 --no-repeat-ngram 4

Cada fragmento arranca solo con  <|file|> <|lang_x|>  y el modelo escribe el archivo. Se informa:
  - estricta:   el fragmento completo parsea;
  - recortada:  tras quitar las lineas finales cortadas por el limite de tokens, parsea y conserva
                >= 80 % de las lineas (separa «mal escrito» de «cortado a mitad»);
  - terminados: solo los que el modelo cerro por si mismo con <|endoftext|>;
  - repeticion: fraccion de lineas que repiten otra del mismo fragmento;
  - Python sin nombres sin definir (pyflakes), entre los fragmentos Python que parsean.
Calibracion: los mismos criterios aplicados a archivos REALES del conjunto de validacion que caben
en el mismo limite de tokens. Si el validador rechazara codigo bueno, se veria ahi.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.eval.harness import Decoding, generate_fragments, score
from src.eval.syntax import RETENTION_MIN, plan_counts, summarize
from src.sample import load_model

PY_LANG, JS_LANG, TS_LANG = "py", "js", "ts"


def real_documents(val_bin: Path, tok, max_tokens: int, per_class: dict, seed: int) -> list[dict]:
    """Archivos completos de val.bin (<|file|> <|lang|> ... <|endoftext|>) de hasta `max_tokens` tokens."""
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    file_id, eot = tok.token_to_id("<|file|>"), tok.token_to_id("<|endoftext|>")
    lang_ids = {tok.token_to_id(f"<|lang_{l}|>"): l for l in (JS_LANG, TS_LANG, PY_LANG)}
    ends = np.flatnonzero(data == eot)
    starts = np.concatenate([[0], ends[:-1] + 1])
    order = np.random.default_rng(seed).permutation(len(ends))
    got: dict[str, list] = {l: [] for l in per_class}
    for i in order:
        s, e = int(starts[i]), int(ends[i])
        if data[s] != file_id or e - s - 3 > max_tokens or e - s < 8:
            continue
        lang = lang_ids.get(int(data[s + 1]))
        if lang and len(got[lang]) < per_class[lang]:
            got[lang].append(tok.decode(data[s + 2:e].astype(np.int64).tolist(), skip_special_tokens=False))
    return [{"id": f"real-{l}-{k}", "lang": l, "code": c} for l, cs in got.items() for k, c in enumerate(cs)]


def python_names_share(rows: list[dict]) -> dict:
    """Entre los fragmentos Python que parsean enteros: cuantos no tienen nombres sin definir."""
    py = [r for r in rows if r["lang"] == "py" and r["ok"]]
    clean = sum(1 for r in py if r["undefined_names"] is False)
    return {"n_parsing_python": len(py), "without_undefined_names": clean / len(py) if py else None}


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def print_table(title: str, s: dict) -> None:
    print(f"\n{title}")
    print(f"{'clase':<7}{'n':>4}{'estricta':>10}{'recortada':>11}{'terminados':>12}{'estricta(term.)':>17}{'retencion':>11}{'repetidas':>11}{'repetitivos':>13}")
    for k in ("total", "js", "ts", "py"):
        if k in s:
            b = s[k]
            print(f"{k:<7}{b['n']:>4}{pct(b['strict']):>10}{pct(b['trimmed']):>11}{b['finished_n']:>12}"
                  f"{pct(b['finished_strict']):>17}{pct(b['mean_retention']):>11}{pct(b['mean_repeat']):>11}"
                  f"{pct(b['repetitive_share']):>13}")


def add_decoding_args(ap: argparse.ArgumentParser) -> None:
    d = Decoding()
    ap.add_argument("--temperature", type=float, default=d.temperature)
    ap.add_argument("--top-k", type=int, default=d.top_k)
    ap.add_argument("--top-p", type=float, default=d.top_p)
    ap.add_argument("--rep-penalty", type=float, default=d.repetition_penalty)
    ap.add_argument("--no-repeat-ngram", type=int, default=d.no_repeat_ngram_size)
    ap.add_argument("--min-p", type=float, default=d.min_p)


def decoding_from_args(args) -> Decoding:
    return Decoding(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    repetition_penalty=args.rep_penalty, no_repeat_ngram_size=args.no_repeat_ngram, min_p=args.min_p)


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="C:/llm-fullstack-data/last_syntax_eval.json", help="los resultados oficiales van a benchmarks/ con --out explicito")
    ap.add_argument("--no-baseline", action="store_true")
    add_decoding_args(ap)
    args = ap.parse_args()
    decoding = decoding_from_args(args)

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    ckpt = args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt"
    model, step = load_model(ckpt, cfg)

    t0 = time.time()
    gen_scored = score(generate_fragments(model, tok, args.n, args.max_new_tokens, decoding, args.seed, progress=True))
    print(f"({time.time() - t0:.0f}s)")
    summary = {"generated": summarize(gen_scored), "python_names": python_names_share(gen_scored)}
    print_table(f"Generados por el modelo (paso {step}; {decoding.label()}; max {args.max_new_tokens} tokens)",
                summary["generated"])
    print(f"Python que parsea y no tiene nombres sin definir: {pct(summary['python_names']['without_undefined_names'])} "
          f"(de {summary['python_names']['n_parsing_python']})")

    real_scored = []
    if not args.no_baseline:
        real = real_documents(cfg.paths.val_bin, tok, args.max_new_tokens, plan_counts(args.n), seed=args.seed)
        real_scored = score([{**r, "finished": True} for r in real])
        summary["real_validation_files"] = summarize(real_scored)
        summary["real_python_names"] = python_names_share(real_scored)
        print_table(f"Calibracion: archivos reales de validacion de <= {args.max_new_tokens} tokens", summary["real_validation_files"])

    out = {"checkpoint": str(ckpt), "checkpoint_step": step, "decoding": decoding.generate_kwargs(),
           "settings": {"n": args.n, "max_new_tokens": args.max_new_tokens, "seed": args.seed},
           "retention_min": RETENTION_MIN, "summary": summary, "fragments": gen_scored, "real_files": real_scored}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
