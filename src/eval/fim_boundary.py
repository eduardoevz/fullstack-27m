"""Diagnostico (Fase 11): ¿el sufijo ayuda poco por un desajuste de tokenizacion entre entrenamiento y evaluacion?

Los tokens de salto de linea incluyen la sangria ("\\n    "). En entrenamiento el corte cae justo despues de un token
con salto, asi que el prefijo termina DENTRO de ese token. En fim_eval el prefijo es texto que acaba en "\\n" y se codifica
aparte, y el medio (con su sangria) tambien. Aqui se mide la misma pregunta con las dos construcciones sobre los mismos
documentos:
  - "texto":  como fim_eval (se corta el texto por lineas y cada trozo se codifica por separado);
  - "token":  como el entrenamiento (se codifica el documento entero y se corta en tokens con salto de linea).

    python -m src.eval.fim_boundary --ckpt /c/llm-fullstack-data/checkpoints_v2_long/ckpt-0007500.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

MAX_PREFIX_TOKENS, MAX_SUFFIX_TOKENS = 220, 120
MIN_PREFIX_LINES, MIN_SUFFIX_LINES = 3, 2


def token_aligned_cases(ids: list, newline_ids, n_lines: int, rng: np.random.Generator):
    """(prefijo, medio, sufijo) en ids: cortes justo despues de tokens con salto de linea, medio de `n_lines` saltos.
    None si el documento es demasiado corto."""
    nl = np.flatnonzero(np.isin(np.asarray(ids), np.asarray(list(newline_ids)))) + 1      # posiciones de corte
    lo, hi = MIN_PREFIX_LINES, len(nl) - n_lines - MIN_SUFFIX_LINES
    if hi < lo:
        return None
    j = int(rng.integers(lo, hi + 1))
    a, b = int(nl[j - 1]), int(nl[j - 1 + n_lines])
    return ids[:a], ids[a:b], ids[b:]


def main() -> None:
    from tokenizers import Tokenizer

    from src.config import REPO_ROOT, load_config
    from src.eval.fim_eval import load_docs, loss_on_middle
    from src.eval.fim_monitor import newline_token_ids
    from src.sample import load_model
    from src.tokenizer.pretokenizer import lang_token

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "model_v2_long.json"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "fim_boundary.json"))
    args = ap.parse_args()

    import torch
    cfg1, cfg2 = load_config(), load_config(args.config)
    torch.set_num_threads(cfg2.training.num_threads)
    tok = Tokenizer.from_file(cfg2.paths.tokenizer_file)
    v1, _ = load_model(Path(cfg1.paths.checkpoint_dir) / "v1-best.pt", cfg1)
    fim, step = load_model(args.ckpt, cfg2)
    nlids = set(newline_token_ids(tok))
    tid = tok.token_to_id
    P, S, M = tid("<|fim_prefix|>"), tid("<|fim_suffix|>"), tid("<|fim_middle|>")
    rng = np.random.default_rng(args.seed)
    docs = load_docs(tok, Path(cfg2.paths.token_dir), per_file=max(1, args.n // 5 + 20), seed=args.seed)
    order = rng.permutation(len(docs))

    res = {"texto": {"fim": [], "v1": []}, "token": {"fim": [], "v1": []}}
    used = 0
    for i in order:
        d = docs[i]
        ids = tok.encode(d["text"]).ids
        head = [tid("<|file|>"), tid(lang_token(d["lang"]))]
        k = int(rng.integers(1, 4))
        tc = token_aligned_cases(ids, nlids, k, rng)
        lines = d["text"].splitlines(keepends=True)
        hi = len(lines) - k - MIN_SUFFIX_LINES
        if tc is None or hi < MIN_PREFIX_LINES:
            continue
        cut = int(rng.integers(MIN_PREFIX_LINES, hi + 1))
        pre_t, mid_t, suf_t = "".join(lines[:cut]), "".join(lines[cut:cut + k]), "".join(lines[cut + k:])
        if not mid_t.strip():
            continue
        variants = {"texto": (tok.encode(pre_t).ids, tok.encode(mid_t).ids, tok.encode(suf_t).ids), "token": tc}
        ok = True
        out = {}
        for name, (pre, mid, suf) in variants.items():
            pre, suf = pre[-MAX_PREFIX_TOKENS:], suf[:MAX_SUFFIX_TOKENS]
            a = loss_on_middle(fim, head + [P] + pre + [S] + suf + [M], mid)
            b = loss_on_middle(v1, head + pre, mid)
            if a is None or b is None:
                ok = False
                break
            out[name] = (a[0], b[0])
        if not ok:
            continue
        for name, (a, b) in out.items():
            res[name]["fim"].append(a)
            res[name]["v1"].append(b)
        used += 1
        if used == args.n:
            break

    summary = {"checkpoint_step": step, "cases": used}
    for name, r in res.items():
        f, v = float(np.mean(r["fim"])), float(np.mean(r["v1"]))
        summary[name] = {"loss_fim": f, "loss_v1_prefix_only": v, "ratio": f / v}
    Path(args.out).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
