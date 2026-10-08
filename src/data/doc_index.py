"""Indice persistente de documentos del corpus v1 (Fase 9, paso 4).

Una sola pasada sobre `raw/`: longitud en tokens (con el mismo encuadre que `encode.py`),
framework, lenguaje, hold-out, repo y el veredicto de las reglas de limpieza. Con eso se
reconstruye, usando `choose_split` (la misma funcion de v1), que documentos entraron en los
500 M tokens de v1 y cuales son sobrantes "frescos" que v1 nunca vio.

    python -m src.data.doc_index
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.encode import N_SPECIAL, choose_split, special_ids
from src.data.filters import clean_document, quality_reject
from src.tokenizer.corpus import is_holdout, iter_docs

CLEAN_REASONS = ["ok", "dts", "template", "python2", "py_syntax", "too_small", "low_alnum", "long_lines", "other"]


def _reason_code(doc, cfg) -> int:
    cleaned, reason = clean_document(doc["code"], doc["path"], doc["lang"])
    if reason is None:
        reason = quality_reject(cleaned, cfg) if cfg is not None else None
    if reason is None:
        return 0
    return CLEAN_REASONS.index(reason) if reason in CLEAN_REASONS else CLEAN_REASONS.index("other")


def index_corpus(raw_dir, tok: Tokenizer, sp, cfg=None, batch_docs: int = 2000, log=None) -> dict:
    cols = {k: [] for k in ("length", "fw", "lang", "hold", "repo", "clean", "seq")}
    batch, seq, t0 = [], 0, time.time()

    def flush():
        encs = tok.encode_batch([d["code"] for _, d in batch], add_special_tokens=False)
        for (s, d), e in zip(batch, encs):
            if len(e.ids) and min(e.ids) < N_SPECIAL:       # igual que _encode_stage: se omite
                continue
            cols["length"].append(len(e.ids) + 3)           # <|file|> <|lang|> ... <|endoftext|>
            cols["fw"].append(d["framework"])
            cols["lang"].append(d["lang"])
            cols["hold"].append(is_holdout(d["repo"]))
            cols["repo"].append(d["repo"])
            cols["clean"].append(_reason_code(d, cfg))
            cols["seq"].append(s)
        batch.clear()

    for d in iter_docs(raw_dir):
        batch.append((seq, d))
        seq += 1
        if len(batch) >= batch_docs:
            flush()
            if log and seq % (batch_docs * 25) == 0:
                log(f"  indexados {seq:,} docs | {time.time() - t0:.0f}s")
    if batch:
        flush()
    return {"length": np.array(cols["length"], dtype=np.int64), "fw": cols["fw"], "lang": cols["lang"],
            "hold": np.array(cols["hold"], dtype=bool), "repo": cols["repo"],
            "clean": np.array(cols["clean"], dtype=np.int8), "seq": np.array(cols["seq"], dtype=np.int64)}


def used_by_v1(idx: dict, train_tokens: int, val_tokens: int, seed: int = 0) -> np.ndarray:
    """Mascara booleana: True para los documentos que entraron en train.bin de v1."""
    _, _, train_idx, _, _, _, _ = choose_split(idx["length"], idx["fw"], idx["hold"], train_tokens, val_tokens,
                                                seed, log=lambda *_: None)
    used = np.zeros(len(idx["length"]), dtype=bool)
    used[train_idx] = True
    return used


def summarize_leftover(idx: dict, used: np.ndarray) -> dict:
    """Sobrantes de train (no hold-out, no usados por v1) por framework, antes y despues de limpiar."""
    left = ~used & ~idx["hold"]
    fws = np.array(idx["fw"])
    out = {}
    for fw in sorted(set(fws[left].tolist())):
        sel = left & (fws == fw)
        clean_ok = sel & (idx["clean"] == 0)
        out[fw] = {"docs": int(sel.sum()), "tokens": int(idx["length"][sel].sum()),
                   "clean_docs": int(clean_ok.sum()), "clean_tokens": int(idx["length"][clean_ok].sum())}
    reasons = {CLEAN_REASONS[c]: int(idx["length"][(idx["clean"] == c)].sum())
               for c in range(1, len(CLEAN_REASONS)) if (idx["clean"] == c).any()}
    return {"leftover_by_framework": out, "rejected_tokens_by_reason_all_corpus": reasons}


def main() -> None:
    from src.config import load_config
    cfg = load_config()
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    sp = special_ids(tok)
    print("Indexando el corpus v1 (tokeniza + limpieza)...", flush=True)
    idx = index_corpus(cfg.paths.raw_dir, tok, sp, cfg.data, log=lambda m: print(m, flush=True))
    used = used_by_v1(idx, cfg.data.target_train_tokens, cfg.data.target_val_tokens, seed=0)

    meta = json.loads((Path(cfg.paths.token_dir) / "meta.json").read_text())
    ok = (int(idx["length"][used].sum()) == meta["train_tokens"] and int(used.sum()) == meta["train_docs"]
          and len(idx["length"]) == meta["docs_encoded"])
    print(f"verificacion contra meta.json de v1: {'OK' if ok else 'FALLA'} "
          f"(tokens {int(idx['length'][used].sum()):,} vs {meta['train_tokens']:,}; docs {int(used.sum()):,} vs {meta['train_docs']:,})")
    if not ok:
        raise SystemExit("la reconstruccion no coincide con v1; no guardo el indice")

    out = Path(cfg.paths.data_root) / "doc_index_v1.npz"
    np.savez_compressed(out, seq=idx["seq"], length=idx["length"], hold=idx["hold"], clean=idx["clean"], used_v1=used,
                        fw=np.array(idx["fw"]), lang=np.array(idx["lang"]), reasons=np.array(CLEAN_REASONS))
    report = {"verified_against_v1_meta": True, **summarize_leftover(idx, used)}
    Path("benchmarks/leftover_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
