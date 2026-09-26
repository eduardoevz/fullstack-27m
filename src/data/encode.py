"""Codifica el corpus a binarios uint16 (train.bin / val.bin) para entrenar con memmap.

Etapa A: tokeniza todo el corpus a un archivo plano temporal (_all.bin) + un indice
         por documento (offset, longitud, framework, lenguaje, hold-out).
Etapa B: elige los tokens de entrenamiento reequilibrando la mezcla SIN duplicar
         archivos (tope comun K por clase, "water-filling"), baraja a nivel de
         documento y escribe train.bin / val.bin leyendo por bloques.

Cada documento queda como  <|file|> <|lang_x|> tokens... <|endoftext|>.
El hold-out (val) es el mismo reparto por repo que uso el tokenizer.

    python src/data/encode.py            # corpus completo
    python src/data/encode.py --max-docs 20000 --out-dir C:/tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.tokenizer.corpus import is_holdout, iter_docs
from src.tokenizer.pretokenizer import SPECIAL_TOKENS, lang_token

N_SPECIAL = len(SPECIAL_TOKENS)


@dataclass(frozen=True)
class SpecialIds:
    file: int
    eot: int
    lang: dict


def special_ids(tok: Tokenizer) -> SpecialIds:
    return SpecialIds(
        file=tok.token_to_id("<|file|>"),
        eot=tok.token_to_id("<|endoftext|>"),
        lang={l: tok.token_to_id(lang_token(l)) for l in ("js", "jsx", "ts", "tsx", "py")},
    )


def frame_doc(ids: np.ndarray, lang: str, sp: SpecialIds) -> np.ndarray:
    head = np.array([sp.file, sp.lang[lang]], dtype=np.uint16)
    tail = np.array([sp.eot], dtype=np.uint16)
    return np.concatenate([head, ids.astype(np.uint16), tail])


def balance_fractions(class_tokens: dict, target: int) -> dict:
    """Fraccion de tokens a conservar por clase con un tope comun K (water-filling).

    Mayor K entero tal que sum(min(tokens_c, K)) <= target. Las clases con
    tokens_c <= K se conservan enteras (fraccion 1.0). Si el total ya cabe, todo entra.
    """
    if sum(class_tokens.values()) <= target:
        return {c: 1.0 for c in class_tokens}
    lo, hi = 0, max(class_tokens.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(t, mid) for t in class_tokens.values()) <= target:
            lo = mid
        else:
            hi = mid - 1
    return {c: min(1.0, lo / t) if t else 1.0 for c, t in class_tokens.items()}


def select_docs(classes: np.ndarray, lengths: np.ndarray, fractions: dict, rng: np.random.Generator) -> np.ndarray:
    """Indices de documentos elegidos: por clase, al azar y sin repetir, hasta fraccion*tokens."""
    taken = []
    for c in sorted(fractions):
        idx = np.flatnonzero(classes == c)
        if not len(idx):
            continue
        if fractions[c] >= 1.0:
            taken.append(idx)
            continue
        budget = fractions[c] * lengths[idx].sum()
        perm = rng.permutation(idx)
        taken.append(perm[np.cumsum(lengths[perm]) <= budget])
    return np.concatenate(taken) if taken else np.empty(0, dtype=np.int64)


def write_split(mm, offsets, lengths, order, out_path: Path, block_docs: int = 4096) -> int:
    total = 0
    with open(out_path, "wb") as f:
        for s in range(0, len(order), block_docs):
            chunk = order[s:s + block_docs]
            block = np.concatenate([mm[offsets[i]:offsets[i] + lengths[i]] for i in chunk])
            f.write(block.astype(np.uint16, copy=False).tobytes())
            total += len(block)
    return total


def _encode_stage(raw_dir, tok, all_path: Path, sp: SpecialIds, max_docs, batch_docs, log):
    offsets, lengths, fws, langs, hold, repos = [], [], [], [], [], []
    state = {"pos": 0, "skipped": 0}
    batch: list = []
    seen = 0
    t0 = time.time()

    def flush(f):
        encs = tok.encode_batch([d["code"] for d in batch], add_special_tokens=False)
        for d, e in zip(batch, encs):
            ids = np.array(e.ids, dtype=np.uint16)
            if len(ids) and int(ids.min()) < N_SPECIAL:   # el texto contenia un token especial literal
                state["skipped"] += 1
                continue
            arr = frame_doc(ids, d["lang"], sp)
            f.write(arr.tobytes())
            offsets.append(state["pos"])
            lengths.append(len(arr))
            fws.append(d["framework"])
            langs.append(d["lang"])
            hold.append(is_holdout(d["repo"]))
            repos.append(d["repo"])
            state["pos"] += len(arr)
        batch.clear()

    with open(all_path, "wb") as f:
        for d in iter_docs(raw_dir):
            batch.append(d)
            seen += 1
            if len(batch) >= batch_docs:
                flush(f)
                if seen % (batch_docs * 25) == 0:
                    log(f"  codificados {seen:,} docs | {state['pos'] / 1e6:.0f} M tokens | {time.time() - t0:.0f}s")
            if max_docs and seen >= max_docs:
                break
        if batch:
            flush(f)
    return (np.array(offsets, dtype=np.int64), np.array(lengths, dtype=np.int64), fws, langs,
            np.array(hold, dtype=bool), repos, state["skipped"])


def build_dataset(raw_dir, tok: Tokenizer, out_dir, train_tokens: int, val_tokens: int, seed: int = 0,
                  max_docs: int | None = None, keep_temp: bool = False, batch_docs: int = 2000,
                  record_repos: bool = False, log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert tok.get_vocab_size() <= 65536, "uint16 no alcanza"
    sp = special_ids(tok)
    all_path = out_dir / "_all.bin"

    log("Etapa A: codificando el corpus...")
    offsets, lengths, fws, langs, hold, repos, skipped = _encode_stage(raw_dir, tok, all_path, sp, max_docs, batch_docs, log)
    names = sorted(set(fws))
    cls = np.array([names.index(f) for f in fws], dtype=np.int64)
    log(f"  {len(lengths):,} docs, {int(lengths.sum()) / 1e6:.1f} M tokens medidos ({skipped} omitidos por contener tokens especiales)")

    log("Etapa B: reequilibrio, barajado y escritura...")
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(~hold)
    before = {names[c]: int(lengths[pool][cls[pool] == c].sum()) for c in range(len(names))}
    frac_by_name = balance_fractions(before, train_tokens)
    frac = {c: frac_by_name[names[c]] for c in range(len(names))}

    train_idx = pool[select_docs(cls[pool], lengths[pool], frac, rng)]
    train_idx = train_idx[rng.permutation(len(train_idx))]
    hold_pool = np.flatnonzero(hold)
    val_idx = hold_pool[select_docs(cls[hold_pool], lengths[hold_pool], frac, rng)]
    val_idx = val_idx[rng.permutation(len(val_idx))]
    val_idx = val_idx[np.cumsum(lengths[val_idx]) <= val_tokens]
    assert not hold[train_idx].any() and hold[val_idx].all()

    mm = np.memmap(all_path, dtype=np.uint16, mode="r")
    n_train = write_split(mm, offsets, lengths, train_idx, out_dir / "train.bin")
    n_val = write_split(mm, offsets, lengths, val_idx, out_dir / "val.bin")
    del mm

    classes = {}
    for c, name in enumerate(names):
        sel = train_idx[cls[train_idx] == c]
        classes[name] = {
            "fraction_kept": frac_by_name[name],
            "train_before_docs": int((cls[pool] == c).sum()), "train_before_tokens": before[name],
            "train_after_docs": int(len(sel)), "train_after_tokens": int(lengths[sel].sum()),
        }
    lang_after: dict = {}
    for i in train_idx:
        lang_after[langs[i]] = lang_after.get(langs[i], 0) + int(lengths[i])
    meta = {
        "vocab_size": tok.get_vocab_size(), "seed": seed, "target_train_tokens": train_tokens, "target_val_tokens": val_tokens,
        "docs_encoded": int(len(lengths)), "docs_skipped_special": skipped, "corpus_tokens_measured": int(lengths.sum()),
        "train_tokens": n_train, "train_docs": int(len(train_idx)), "val_tokens": n_val, "val_docs": int(len(val_idx)),
        "classes": classes, "train_tokens_by_lang": lang_after,
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-docs", type=int, default=None, help="solo para pruebas de humo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-temp", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    if tok.get_vocab_size() != cfg.model.vocab_size:
        raise SystemExit(f"tokenizer ({tok.get_vocab_size()}) != vocab_size del config ({cfg.model.vocab_size})")
    out = args.out_dir or cfg.paths.token_dir
    meta = build_dataset(cfg.paths.raw_dir, tok, out, cfg.data.target_train_tokens, cfg.data.target_val_tokens,
                         seed=args.seed, max_docs=args.max_docs, keep_temp=args.keep_temp)
    print(f"\ntrain: {meta['train_tokens'] / 1e6:.1f} M tokens ({meta['train_docs']:,} docs) | "
          f"val: {meta['val_tokens'] / 1e6:.2f} M ({meta['val_docs']:,} docs)")
    print(f"corpus medido: {meta['corpus_tokens_measured'] / 1e6:.1f} M tokens | RSS final {meta['rss_gb_at_end']} GB")


if __name__ == "__main__":
    main()
