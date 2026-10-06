"""Metrica de validez sintactica: genera N fragmentos desde cero y mide que porcentaje parsea.

    python src/eval/syntax_check.py --n 100
    python src/eval/syntax_check.py --n 100 --ckpt C:/llm-fullstack-data/checkpoints/best.pt

Cada fragmento arranca solo con  <|file|> <|lang_x|>  y el modelo escribe el archivo. Se informa:
  - estricta:   el fragmento completo parsea;
  - recortada:  tras quitar las lineas finales cortadas por el limite de tokens, parsea y conserva
                >= 80 % de las lineas (separa «mal escrito» de «cortado a mitad»);
  - terminados: solo los que el modelo cerro por si mismo con <|endoftext|>.
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
from src.eval.syntax import RETENTION_MIN, check_fragments, plan_counts, repeated_line_ratio, summarize, trim_to_valid
from src.sample import generate, load_model

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


def score(rows_in: list[dict]) -> list[dict]:
    """rows_in: [{id, lang, code, finished}] -> filas con ok / trimmed_ok / retention."""
    verdicts = check_fragments([{"id": r["id"], "lang": r["lang"], "code": r["code"]} for r in rows_in])
    rows = []
    for r, v in zip(rows_in, verdicts):
        if v["ok"]:
            trimmed_ok, retention = True, 1.0
        else:
            kept, retention = trim_to_valid(r["code"], r["lang"])
            trimmed_ok = retention >= RETENTION_MIN
        rows.append({**r, "ok": v["ok"], "error": v["error"], "trimmed_ok": trimmed_ok, "retention": retention,
                     "repeat": repeated_line_ratio(r["code"])})
    return rows


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


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "syntax_eval.json"))
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    ckpt = args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt"
    model, step = load_model(ckpt, cfg)
    eot = tok.token_to_id("<|endoftext|>")
    head = {l: [tok.token_to_id("<|file|>"), tok.token_to_id(f"<|lang_{l}|>")] for l in (JS_LANG, TS_LANG, PY_LANG)}

    gen_rows, t0, i = [], time.time(), 0
    for lang, count in plan_counts(args.n).items():
        for _ in range(count):
            g = torch.Generator().manual_seed(args.seed + i)
            new = generate(model, head[lang], args.max_new_tokens, args.temperature, args.top_k, args.top_p, eot, g)
            gen_rows.append({"id": f"gen-{lang}-{i}", "lang": lang, "finished": len(new) < args.max_new_tokens,
                             "n_tokens": len(new), "code": tok.decode(new, skip_special_tokens=False)})
            i += 1
            print(f"\rgenerados {i}/{args.n}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    gen_scored = score(gen_rows)
    summary = {"generated": summarize(gen_scored)}
    print_table(f"Generados por el modelo (paso {step}, T={args.temperature}, top-k {args.top_k}, top-p {args.top_p}, "
                f"max {args.max_new_tokens} tokens)", summary["generated"])

    real_scored = []
    if not args.no_baseline:
        real = real_documents(cfg.paths.val_bin, tok, args.max_new_tokens, plan_counts(args.n), seed=args.seed)
        real_scored = score([{**r, "finished": True} for r in real])
        summary["real_validation_files"] = summarize(real_scored)
        print_table(f"Calibracion: archivos reales de validacion de <= {args.max_new_tokens} tokens", summary["real_validation_files"])

    out = {"checkpoint_step": step, "settings": {k: getattr(args, k) for k in
           ("n", "max_new_tokens", "temperature", "top_k", "top_p", "seed")},
           "retention_min": RETENTION_MIN, "summary": summary, "fragments": gen_scored, "real_files": real_scored}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
