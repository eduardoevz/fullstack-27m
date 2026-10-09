"""Evaluacion de infilling (Fase 10). Reglas fijadas antes de medir (ver docs/plan-mejoras-modelo.md).

Casos: documentos de validacion (hold-out por repo) con un MEDIO de 1-3 lineas. Se compara:
  - v1 solo con prefijo (no conoce FIM)  frente a  el modelo FIM con prefijo + sufijo;
  - perdida teacher-forced del medio, coincidencia exacta/similitud del medio generado (T=0) y
    % de archivos reconstruidos (prefijo + generado + sufijo) que parsean.

    python -m src.eval.fim_eval --n 200
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

MAX_PREFIX_TOKENS, MAX_SUFFIX_TOKENS = 220, 120
MIN_PREFIX_LINES, MIN_SUFFIX_LINES = 3, 2


def build_cases(docs: list, n: int, seed: int) -> list:
    """docs: [{lang, text}]. Un caso por documento (en orden aleatorio) hasta `n`; omite los cortos."""
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(docs)):
        lines = docs[i]["text"].splitlines(keepends=True)
        k = int(rng.integers(1, 4))
        lo, hi = MIN_PREFIX_LINES, len(lines) - k - MIN_SUFFIX_LINES
        if hi < lo:
            continue
        cut = int(rng.integers(lo, hi + 1))
        mid = "".join(lines[cut:cut + k])
        if not mid.strip() or not mid.endswith("\n"):
            continue
        out.append({"lang": docs[i]["lang"], "prefix_full": "".join(lines[:cut]), "middle": mid,
                    "suffix_full": "".join(lines[cut + k:])})
        if len(out) == n:
            break
    return out


def trim_context(prefix: str, suffix: str, count, max_prefix: int, max_suffix: int):
    """Recorta lineas por el lado LEJANO al cursor hasta que cada lado cabe en su presupuesto de tokens."""
    pl, sl = prefix.splitlines(keepends=True), suffix.splitlines(keepends=True)
    while pl and count("".join(pl)) > max_prefix:
        pl.pop(0)
    while sl and count("".join(sl)) > max_suffix:
        sl.pop()
    return "".join(pl), "".join(sl)


def middle_similarity(generated: str, true: str) -> dict:
    g, t = generated.strip(), true.strip()
    return {"exact": float(g == t), "similarity": difflib.SequenceMatcher(None, g, t).ratio()}


@torch.no_grad()
def loss_on_middle(model, prompt_ids: list, mid_ids: list):
    """(perdida media, n tokens) de los tokens del medio dado el prompt; None si no cabe en el contexto."""
    if len(prompt_ids) + len(mid_ids) > model.cfg.block_size or not mid_ids:
        return None
    ids = torch.tensor([prompt_ids + mid_ids])
    logits, _ = model(ids)
    start = len(prompt_ids) - 1
    tgt = torch.tensor(mid_ids)
    loss = F.cross_entropy(logits[0, start:start + len(mid_ids)], tgt, reduction="mean")
    return float(loss), len(mid_ids)


def load_docs(tok, token_dir: Path, per_file: int, seed: int) -> list:
    """Documentos completos de val.bin y de los val_<dominio>.bin, decodificados a texto."""
    from src.training.fim import doc_bounds
    file_id, eot = tok.token_to_id("<|file|>"), tok.token_to_id("<|endoftext|>")
    lang_of = {tok.token_to_id(f"<|lang_{l}|>"): l for l in ("js", "ts", "py")}
    docs = []
    for p in sorted(token_dir.glob("val*.bin")):
        data = np.memmap(p, dtype=np.uint16, mode="r")
        starts, ends = doc_bounds(data, file_id, eot)
        pick = np.random.default_rng(seed).permutation(len(starts))[: per_file * 6]
        got = 0
        for i in pick:
            s, e = int(starts[i]), int(ends[i])
            if e - s < 60 or e - s > 700 or int(data[s + 1]) not in lang_of:
                continue
            docs.append({"lang": lang_of[int(data[s + 1])], "set": p.stem,
                         "text": tok.decode(data[s + 2:e].tolist(), skip_special_tokens=False)})
            got += 1
            if got == per_file:
                break
    return docs


def main() -> None:
    from tokenizers import Tokenizer

    from src.config import REPO_ROOT, load_config
    from src.eval.syntax import check_fragments
    from src.sample import frame_fim_prompt, frame_prompt, generate, load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config-fim", default=str(REPO_ROOT / "config" / "model_v2.json"))
    ap.add_argument("--ckpt-fim", default=None, help="por defecto best.pt de la carpeta del config FIM")
    ap.add_argument("--ckpt-v1", default=None, help="por defecto v1-best.pt")
    ap.add_argument("--trim-fim", action="store_true",
                    help="analisis posterior: recorta tambien la salida FIM al numero de lineas del medio verdadero")
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "fim_eval.json"))
    args = ap.parse_args()

    cfg1, cfg2 = load_config(), load_config(args.config_fim)
    torch.set_num_threads(cfg2.training.num_threads)
    tok = Tokenizer.from_file(cfg2.paths.tokenizer_file)        # mismos ids para texto normal en v1 y v2
    v1, _ = load_model(args.ckpt_v1 or Path(cfg1.paths.checkpoint_dir) / "v1-best.pt", cfg1)
    fim, step = load_model(args.ckpt_fim or Path(cfg2.paths.checkpoint_dir) / "best.pt", cfg2)
    eot = tok.token_to_id("<|endoftext|>")
    count = lambda s: len(tok.encode(s).ids)

    docs = load_docs(tok, Path(cfg2.paths.token_dir), per_file=max(1, args.n // 5 + 20), seed=args.seed)
    cases = build_cases(docs, n=args.n * 2, seed=args.seed)
    # Solo cuentan los archivos originales que parsean (si no, el validador no distingue).
    oks = check_fragments([{"id": i, "lang": c["lang"], "code": c["prefix_full"] + c["middle"] + c["suffix_full"]}
                           for i, c in enumerate(cases)])
    cases = [c for c, r in zip(cases, oks) if r["ok"]][: args.n]
    print(f"{len(cases)} casos (originales que parsean)", flush=True)

    rows = []
    for i, c in enumerate(cases):
        pre, suf = trim_context(c["prefix_full"], c["suffix_full"], count, MAX_PREFIX_TOKENS, MAX_SUFFIX_TOKENS)
        mid_ids = tok.encode(c["middle"]).ids
        r = {"lang": c["lang"], "n_mid": len(mid_ids)}
        a = loss_on_middle(v1, frame_prompt(tok, pre, c["lang"]), mid_ids)
        b = loss_on_middle(fim, frame_fim_prompt(tok, pre, suf, c["lang"]), mid_ids)
        r["loss_v1_prefix"], r["loss_fim"] = (a[0] if a else None), (b[0] if b else None)
        budget = min(80, 2 * len(mid_ids) + 8)
        g_fim = tok.decode(generate(fim, frame_fim_prompt(tok, pre, suf, c["lang"]), budget, 0.0, eos_id=eot),
                           skip_special_tokens=False)
        if args.trim_fim:
            g_fim = "".join(g_fim.splitlines(keepends=True)[: c["middle"].count("\n")])
        g_v1 = tok.decode(generate(v1, frame_prompt(tok, pre, c["lang"]), budget, 0.0, eos_id=eot),
                          skip_special_tokens=False)
        g_v1 = "".join(g_v1.splitlines(keepends=True)[: c["middle"].count("\n")])      # v1 no sabe cuando parar
        recon = {}
        for name, g in (("fim", g_fim), ("v1", g_v1)):
            r[f"{name}_gen"] = g
            r.update({f"{name}_{k}": v for k, v in middle_similarity(g, c["middle"]).items()})
            recon[name] = c["prefix_full"] + (g if g.endswith("\n") else g + "\n") + c["suffix_full"]
        r["_recon"] = recon
        rows.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(cases)}", flush=True)

    for name in ("fim", "v1"):
        res = check_fragments([{"id": i, "lang": r["lang"], "code": r["_recon"][name]} for i, r in enumerate(rows)])
        for r, x in zip(rows, res):
            r[f"{name}_parses"] = float(x["ok"])

    def mean(key, subset=None):
        v = [r[key] for r in (subset if subset is not None else rows) if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    both = [r for r in rows if r["loss_fim"] is not None and r["loss_v1_prefix"] is not None]
    summary = {"checkpoint_fim_step": step, "cases": len(rows), "cases_with_loss": len(both),
               "loss_v1_prefix_only": mean("loss_v1_prefix", both), "loss_fim": mean("loss_fim", both),
               "loss_ratio": (mean("loss_fim", both) / mean("loss_v1_prefix", both)) if both else None,
               "exact_fim": mean("fim_exact"), "exact_v1": mean("v1_exact"),
               "similarity_fim": mean("fim_similarity"), "similarity_v1": mean("v1_similarity"),
               "parses_fim": mean("fim_parses"), "parses_v1": mean("v1_parses")}
    for r in rows:
        r.pop("_recon")
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
