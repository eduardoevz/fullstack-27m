"""Puntuacion de documentos con v1 y corte de la cola peor por dominio (Fase 9, paso 5).

Regla fijada antes de medir: el puntaje de un documento es la perdida media por token sobre
sus primeros `max_len` tokens (sin contar la cabecera `<|file|> <|lang|>`). Se descarta el
`pct` % peor DE CADA framework, para no expulsar un dominio raro solo por serlo para v1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pad_batch(docs, max_len: int, pad_id: int):
    """Trunca a `max_len` y rellena. Devuelve (filas (N, max_len), longitudes reales (N,))."""
    rows = np.full((len(docs), max_len), pad_id, dtype=np.int64)
    lens = np.zeros(len(docs), dtype=np.int64)
    for i, d in enumerate(docs):
        d = d[:max_len]
        rows[i, :len(d)] = d
        lens[i] = len(d)
    return rows, lens


@torch.no_grad()
def doc_mean_losses(model, docs, max_len: int = 256, batch: int = 32, pad_id: int = 0, skip_header: int = 2) -> np.ndarray:
    """Perdida media por token de cada documento (NaN si no queda ningun token objetivo)."""
    out = np.full(len(docs), np.nan, dtype=np.float64)
    for s in range(0, len(docs), batch):
        rows, lens = pad_batch(docs[s:s + batch], max_len, pad_id)
        t = torch.from_numpy(rows)
        logits, _ = model(t[:, :-1])
        loss = F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction="none")   # (B, T-1); objetivo j = token j+1
        pos = torch.arange(1, t.shape[1])[None, :]
        valid = (pos >= skip_header) & (pos < torch.from_numpy(lens)[:, None])
        n = valid.sum(1)
        mean = (loss * valid).sum(1) / n.clamp(min=1)
        mean[n == 0] = float("nan")
        out[s:s + len(rows)] = mean.numpy()
    return out


def tail_thresholds(scores: np.ndarray, groups: np.ndarray, pct: float) -> dict:
    """Umbral por grupo: el percentil (100 - pct) de sus puntajes. Se descarta lo que lo supere."""
    th = {}
    for g in sorted(set(groups.tolist())):
        s = scores[(groups == g) & ~np.isnan(scores)]
        if len(s):
            th[g] = float(np.percentile(s, 100 - pct))
    return th


def worst_tail_mask(scores: np.ndarray, groups: np.ndarray, thresholds: dict) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    for g, t in thresholds.items():
        mask |= (groups == g) & (scores > t)
    return mask


def main() -> None:
    """Calibracion: puntua una muestra estratificada del corpus v1 y propone los umbrales."""
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    from tokenizers import Tokenizer

    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    from src.config import load_config
    from src.data.encode import frame_doc, special_ids
    from src.sample import load_model
    from src.tokenizer.corpus import iter_docs

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--pct", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--review", type=int, default=40, help="documentos peores a volcar para revision manual")
    ap.add_argument("--out", default=str(root / "benchmarks" / "doc_ppl_calibration.json"))
    args = ap.parse_args()

    cfg = load_config()
    z = np.load(Path(cfg.paths.data_root) / "doc_index_v1.npz")
    fw, ok = z["fw"], (z["clean"] == 0) & ~z["hold"]
    rng = np.random.default_rng(args.seed)
    chosen = []
    for f in sorted(set(fw.tolist())):
        cand = np.flatnonzero(ok & (fw == f))
        chosen += rng.choice(cand, size=min(args.per_class, len(cand)), replace=False).tolist()
    wanted = {int(z["seq"][i]): i for i in chosen}

    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    sp = special_ids(tok)
    docs, meta, snippets = [], [], []
    print(f"leyendo {len(wanted)} documentos del corpus...", flush=True)
    for seq, d in enumerate(iter_docs(cfg.paths.raw_dir)):
        if seq in wanted:
            ids = np.array(tok.encode(d["code"], add_special_tokens=False).ids[:args.max_len], dtype=np.uint16)
            docs.append(frame_doc(ids, d["lang"], sp).tolist())
            meta.append((d["framework"], d["lang"], d["repo"], d["path"]))
            snippets.append(d["code"][:400])
    model, _ = load_model(Path(cfg.paths.checkpoint_dir) / "v1-best.pt", cfg)
    t0 = time.time()
    scores = doc_mean_losses(model, docs, max_len=args.max_len + 3, batch=32, pad_id=sp.eot)
    print(f"puntuados {len(docs)} docs en {time.time() - t0:.0f}s", flush=True)

    groups = np.array([m[0] for m in meta])
    th = tail_thresholds(scores, groups, args.pct)
    mask = worst_tail_mask(scores, groups, th)
    per = {}
    for g in sorted(th):
        s = scores[(groups == g) & ~np.isnan(scores)]
        per[g] = {"n": int(len(s)), "p05": float(np.percentile(s, 5)), "median": float(np.median(s)),
                  "p95": float(np.percentile(s, 95)), "threshold": th[g]}
    langs = np.array([m[1] for m in meta])
    tail_by_lang = {l: round(float(mask[langs == l].mean()), 3) for l in sorted(set(langs.tolist()))}
    repos = [m[2] for m, k in zip(meta, mask) if k]
    report = {"per_class": args.per_class, "max_len": args.max_len, "pct": args.pct, "n_scored": int(len(docs)),
              "by_framework": per, "tail_share_by_lang": tail_by_lang,
              "tail_distinct_repos": len(set(repos)), "tail_docs": int(mask.sum()),
              "tokens_scored": int(sum(min(len(d), args.max_len + 3) for d in docs))}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    order = np.argsort(-np.nan_to_num(scores, nan=-1))[:args.review]
    with open(root / "logs" / "doc_ppl_worst.txt", "w", encoding="utf-8") as f:
        for i in order:
            f.write(f"=== loss={scores[i]:.2f} fw={meta[i][0]} lang={meta[i][1]} path={meta[i][3]}\n{snippets[i]}\n\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
