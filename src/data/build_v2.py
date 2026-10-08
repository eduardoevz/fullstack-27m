"""Construccion del dataset v2 (Fase 9, paso 6): corpus v1 limpio + fuente moderna, con pisos de mezcla.

Respecto a `encode.build_dataset`:
- Une varios directorios `raw` (v1 y `raw_modern`) y vuelve a aplicar la limpieza completa
  (`clean_document`: ruido + cabeceras de licencia) antes de codificar.
- La mezcla usa clases (framework, tsx/otro) y *pisos* por dominio (Next.js, FastAPI, tsx):
  `plan_mix` sube el peso de las clases con piso hasta cumplirlo, sin duplicar documentos.
- Dentro de cada clase prefiere documentos "frescos" (que v1 no vio) antes que repetidos.
- Escribe ademas `val_<dominio>.bin` (hold-out por repo) y `train_docs.npz` (offset/longitud de
  cada documento en train.bin, para alinear ventanas en la Fase 10).

    python -m src.data.build_v2 --out-dir C:/llm-fullstack-data/tokens_v2 --train-tokens 600000000
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import psutil
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.encode import _encode_stage, select_docs, special_ids, write_split
from src.data.filters import clean_document, quality_reject

DEFAULT_FLOORS = {"nextjs": 0.05, "fastapi": 0.03, "tsx": 0.15}
DOMAIN_VAL = ("nextjs", "fastapi", "nestjs", "react")


def class_key(framework: str, lang: str) -> str:
    return f"{framework}|{'tsx' if lang == 'tsx' else 'other'}"


def _in_floor(name: str, key: str) -> bool:
    return key.endswith("|tsx") if name == "tsx" else key.startswith(name + "|")


def plan_mix(class_tokens: dict, target: float, floors: dict) -> dict:
    """Fraccion a conservar por clase. Sin pisos equivale a `balance_fractions` (tope comun K).

    Con pisos, las clases que pertenecen a algun grupo con piso pesan `b` veces en el tope
    (tokens_c = min(t_c, K * w_c)); se busca el menor `b` que cumple todos los pisos. Un piso
    inalcanzable (no hay material) se informa en `unmet` y la clase queda entera.
    """
    if sum(class_tokens.values()) <= target:
        return {"fractions": {c: 1.0 for c in class_tokens}, "unmet": {}, "boost": 1.0}
    keys = sorted(class_tokens)
    t = np.array([class_tokens[k] for k in keys], dtype=np.float64)

    def members(n):
        return np.array([_in_floor(n, k) for k in keys])

    # Un piso inalcanzable (todo el grupo no llega) deja ese grupo entero y no refuerza a los demas.
    impossible = {n for n in floors if t[members(n)].sum() < floors[n] * target - 1e-9}
    forced = np.zeros(len(keys), dtype=bool)
    for n in impossible:
        forced |= members(n)
    live = {n: f for n, f in floors.items() if n not in impossible}
    grouped = np.zeros(len(keys), dtype=bool)
    for n in live:
        grouped |= members(n)
    grouped &= ~forced
    free = ~forced
    rest = max(0.0, target - t[forced].sum())

    def kept_for(b):
        w = np.where(grouped, b, 1.0)
        lo, hi = 0.0, float(t.max())
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.minimum(t, mid * w)[free].sum() <= rest:
                lo = mid
            else:
                hi = mid
        return np.where(forced, t, np.minimum(t, lo * w))

    def shares(kept, names):
        return {n: float(kept[members(n)].sum() / kept.sum()) for n in names}

    def ok(kept):
        return all(s >= live[n] - 1e-3 for n, s in shares(kept, live).items())

    kept = kept_for(1.0)
    b = 1.0
    if live and not ok(kept):
        lo, hi = 1.0, 1000.0
        if not ok(kept_for(hi)):
            b, kept = hi, kept_for(hi)
        else:
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(kept_for(mid)) else (mid, hi)
            b, kept = hi, kept_for(hi)
    unmet = {n: s for n, s in shares(kept, floors).items() if s < floors[n] - 1e-3}
    return {"fractions": {k: float(min(1.0, kept[i] / t[i])) if t[i] else 1.0 for i, k in enumerate(keys)},
            "unmet": unmet, "boost": b}


def select_docs_prefer(classes, lengths, fractions, rng, fresh):
    """Como `select_docs`, pero dentro de cada clase entran primero los documentos frescos."""
    taken = []
    for c in sorted(fractions):
        idx = np.flatnonzero(classes == c)
        if not len(idx):
            continue
        if fractions[c] >= 1.0:
            taken.append(idx)
            continue
        budget = fractions[c] * lengths[idx].sum()
        f, u = idx[fresh[idx]], idx[~fresh[idx]]
        perm = np.concatenate([rng.permutation(f), rng.permutation(u)])
        taken.append(perm[np.cumsum(lengths[perm]) <= budget])
    return np.concatenate(taken) if taken else np.empty(0, dtype=np.int64)


def iter_clean_docs(raw_dirs, fresh_by_source, quality_cfg, stats):
    for si, d in enumerate(raw_dirs):
        fresh = fresh_by_source.get(si)
        seq = -1
        for shard in sorted(Path(d).glob("*.jsonl.gz")):
            with gzip.open(shard, "rt", encoding="utf-8", newline="") as f:
                for line in f:
                    seq += 1
                    doc = json.loads(line)
                    cleaned, reason = clean_document(doc["code"], doc["path"], doc["lang"])
                    if reason is None and quality_cfg is not None:
                        reason = quality_reject(cleaned, quality_cfg)
                    if reason:
                        stats[f"rej_{reason}"] = stats.get(f"rej_{reason}", 0) + 1
                        continue
                    doc["code"] = cleaned
                    doc["source"] = si
                    doc["seq"] = seq
                    doc["fresh"] = bool(fresh is not None and seq < len(fresh) and fresh[seq])
                    yield doc


def build_v2(raw_dirs, tok: Tokenizer, out_dir, train_tokens: int, val_tokens: int, floors: dict | None = None,
             fresh_by_source: dict | None = None, domain_val_tokens: int = 1_500_000, seed: int = 0,
             quality_cfg=None, batch_docs: int = 2000, keep_temp: bool = False, record_repos: bool = False,
             log=print) -> dict:
    floors = DEFAULT_FLOORS if floors is None else floors
    fresh_by_source = fresh_by_source or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert tok.get_vocab_size() <= 65536, "uint16 no alcanza"
    sp = special_ids(tok)
    all_path = out_dir / "_all.bin"

    log("Etapa A: limpiando y codificando...")
    rejects: dict = {}
    extras = {"fresh": [], "source": [], "seq": []}
    docs = iter_clean_docs(raw_dirs, fresh_by_source, quality_cfg, rejects)
    offsets, lengths, fws, langs, hold, repos, skipped = _encode_stage(
        None, tok, all_path, sp, None, batch_docs, log, docs=docs, extras=extras)
    fresh = np.array(extras["fresh"], dtype=bool)
    source = np.array(extras["source"], dtype=np.int8)
    log(f"  {len(lengths):,} docs, {int(lengths.sum()) / 1e6:.1f} M tokens; rechazados por limpieza: {rejects}")

    log("Etapa B: mezcla con pisos, barajado y escritura...")
    keys = np.array([class_key(f, l) for f, l in zip(fws, langs)])
    names = sorted(set(keys.tolist()))
    cls = np.array([names.index(k) for k in keys], dtype=np.int64)
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(~hold)
    before = {n: int(lengths[pool][cls[pool] == i].sum()) for i, n in enumerate(names)}
    plan = plan_mix(before, train_tokens, floors)
    frac = {i: plan["fractions"][n] for i, n in enumerate(names)}

    train_idx = pool[select_docs_prefer(cls[pool], lengths[pool], frac, rng, fresh[pool])]
    train_idx = train_idx[rng.permutation(len(train_idx))]
    hold_pool = np.flatnonzero(hold)
    val_idx = hold_pool[select_docs(cls[hold_pool], lengths[hold_pool], frac, rng)]
    val_idx = val_idx[rng.permutation(len(val_idx))]
    val_idx = val_idx[np.cumsum(lengths[val_idx]) <= val_tokens]
    assert not hold[train_idx].any() and hold[val_idx].all()

    fw_arr = np.array(fws)
    domain_idx = {}
    for name in DOMAIN_VAL:
        if domain_val_tokens <= 0:
            break
        cand = hold_pool[fw_arr[hold_pool] == name]
        cand = cand[rng.permutation(len(cand))]
        cand = cand[np.cumsum(lengths[cand]) <= domain_val_tokens]
        if len(cand):
            domain_idx[name] = cand

    mm = np.memmap(all_path, dtype=np.uint16, mode="r")
    n_train = write_split(mm, offsets, lengths, train_idx, out_dir / "train.bin")
    n_val = write_split(mm, offsets, lengths, val_idx, out_dir / "val.bin")
    domain_val = {n: write_split(mm, offsets, lengths, ix, out_dir / f"val_{n}.bin") for n, ix in domain_idx.items()}
    del mm

    tl = lengths[train_idx]
    np.savez_compressed(out_dir / "train_docs.npz", offset=np.concatenate([[0], np.cumsum(tl)[:-1]]).astype(np.int64),
                        length=tl, fw=fw_arr[train_idx], lang=np.array(langs)[train_idx],
                        source=source[train_idx], seq=np.array(extras["seq"], dtype=np.int64)[train_idx],
                        fresh=fresh[train_idx])

    def share_by(values):
        tot = int(tl.sum()) or 1
        out: dict = {}
        for v, ln in zip(values[train_idx], tl):
            out[v] = out.get(v, 0) + int(ln)
        return {k: round(v / tot, 4) for k, v in sorted(out.items())}

    meta = {
        "seed": seed, "target_train_tokens": train_tokens, "target_val_tokens": val_tokens, "floors": floors,
        "docs_encoded": int(len(lengths)), "docs_skipped_special": skipped, "cleaning_rejects": rejects,
        "corpus_tokens_measured": int(lengths.sum()),
        "train_tokens": n_train, "train_docs": int(len(train_idx)), "val_tokens": n_val, "val_docs": int(len(val_idx)),
        "domain_val": domain_val,
        "mix": {"boost": plan["boost"], "unmet_floors": plan["unmet"],
                "share_by_framework": share_by(fw_arr), "share_by_lang": share_by(np.array(langs)),
                "pool_tokens_by_class": before,
                "kept_tokens_by_class": {n: int(lengths[train_idx][cls[train_idx] == i].sum()) for i, n in enumerate(names)}},
        "sources": {"v1_docs": int((source[train_idx] == 0).sum()), "modern_docs": int((source[train_idx] == 1).sum()),
                    "fresh_token_share": round(float(tl[fresh[train_idx]].sum() / max(1, tl.sum())), 4)},
        "rss_gb_at_end": round(psutil.Process().memory_info().rss / 1e9, 2),
    }
    if record_repos:
        meta["train_repos"] = sorted({repos[i] for i in train_idx})
        meta["val_repos"] = sorted({repos[i] for i in val_idx})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if not keep_temp:
        all_path.unlink()
    return meta


def main() -> None:
    from src.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--train-tokens", type=int, default=600_000_000)
    ap.add_argument("--val-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    root = Path(cfg.paths.data_root)
    raw_dirs = [Path(cfg.paths.raw_dir), root / "raw_modern"]
    z = np.load(root / "doc_index_v1.npz")      # marca de lo que v1 ya vio (indice en orden de `seq`)
    fresh = np.zeros(int(z["seq"].max()) + 1, dtype=bool)
    fresh[z["seq"][~z["used_v1"]]] = True
    out = args.out_dir or str(root / "tokens_v2")
    meta = build_v2(raw_dirs, tok, out, args.train_tokens, args.val_tokens, floors=DEFAULT_FLOORS,
                    fresh_by_source={0: fresh}, seed=args.seed, quality_cfg=cfg.data, keep_temp=args.keep_temp)
    print(json.dumps({k: meta[k] for k in ("train_tokens", "train_docs", "val_tokens", "domain_val", "mix", "sources")}, indent=1))


if __name__ == "__main__":
    main()
