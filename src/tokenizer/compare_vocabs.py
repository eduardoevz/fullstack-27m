"""Compara tokenizers (16k vs 32k) sobre el hold-out y decide con la regla del plan.

Metrica de decision: bytes_por_token x tokens_por_segundo_del_modelo. La cabeza de
salida hace que el modelo con vocab 16k corra mas rapido (medido en la Fase 0):
32k = 1057 tok/s, 16k = 1311 tok/s.

    python src/tokenizer/compare_vocabs.py --max-docs 5000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.tokenizer.corpus import iter_holdout

MODEL_TOKS_PER_S = {16384: 1311.0, 32768: 1057.0}
INDENTS = {"2 esp": "\n  ", "4 esp": "\n    ", "8 esp": "\n        ", "12 esp": "\n            ", "tab": "\n\t", "2 tabs": "\n\t\t"}
SPECIAL_PER_DOC = 3  # <|file|>, <|lang_x|> y <|endoftext|>


def evaluate(tok: Tokenizer, docs: list[dict], roundtrip_n: int) -> dict:
    per_lang = defaultdict(lambda: [0, 0, 0])  # bytes, tokens, docs
    t0 = time.time()
    encoded = tok.encode_batch([d["code"] for d in docs])
    enc_time = time.time() - t0
    for d, e in zip(docs, encoded):
        row = per_lang[d["lang"]]
        row[0] += len(d["code"].encode("utf-8"))
        row[1] += len(e.ids)
        row[2] += 1
    tb = sum(r[0] for r in per_lang.values())
    tt = sum(r[1] for r in per_lang.values())
    failures = [i for i, (d, e) in enumerate(zip(docs[:roundtrip_n], encoded)) if tok.decode(e.ids) != d["code"]]
    longest = sorted(tok.get_vocab(), key=len, reverse=True)[:25]
    return {
        "vocab_size": tok.get_vocab_size(),
        "bytes_per_token": tb / tt,
        "tokens_per_doc": tt / len(docs),
        "by_lang": {k: {"bytes_per_token": v[0] / v[1], "tokens_per_doc": v[1] / v[2], "docs": v[2]} for k, v in sorted(per_lang.items())},
        "indent_tokens": {k: len(tok.encode(s).ids) for k, s in INDENTS.items()},
        "roundtrip_checked": min(roundtrip_n, len(docs)),
        "roundtrip_failures": len(failures),
        "encode_mb_per_s": tb / 1e6 / enc_time,
        "longest_tokens": longest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-docs", type=int, default=5000)
    ap.add_argument("--roundtrip-n", type=int, default=1000)
    ap.add_argument("--vocabs", type=int, nargs="+", default=[16384, 32768])
    args = ap.parse_args()

    cfg = load_config(args.config)
    docs = list(iter_holdout(cfg.paths.raw_dir, args.max_docs))
    manifest = json.loads((Path(cfg.paths.raw_dir) / "manifest.json").read_text())["totals"]
    corpus_bytes, corpus_docs = manifest["kept_bytes"], manifest["kept"]
    print(f"hold-out: {len(docs)} documentos, {sum(len(d['code'].encode()) for d in docs) / 1e6:.1f} MB")

    results = {}
    for v in args.vocabs:
        tok = Tokenizer.from_file(str(Path(cfg.paths.data_root) / f"tokenizer-{v}.json"))
        r = evaluate(tok, docs, args.roundtrip_n)
        r["est_corpus_tokens"] = int(corpus_bytes / r["bytes_per_token"] + SPECIAL_PER_DOC * corpus_docs)
        r["decision_score"] = r["bytes_per_token"] * MODEL_TOKS_PER_S[v]
        results[v] = r
        print(f"\n== vocab {v}")
        print(f"  bytes/token {r['bytes_per_token']:.3f} | tokens/doc {r['tokens_per_doc']:.0f} | codificacion {r['encode_mb_per_s']:.1f} MB/s")
        print(f"  round-trip: {r['roundtrip_failures']} fallos de {r['roundtrip_checked']}")
        print(f"  tokens por sangria: {r['indent_tokens']}")
        for lang, s in r["by_lang"].items():
            print(f"  {lang:<4} {s['bytes_per_token']:.3f} B/tok  ({s['docs']} docs)")
        print(f"  tokens estimados del corpus completo: {r['est_corpus_tokens'] / 1e6:.0f} M")
        print(f"  tokens mas largos: {ascii(r['longest_tokens'][:12])}")

    if 16384 in results and 32768 in results:
        rel = results[16384]["bytes_per_token"] / results[32768]["bytes_per_token"]
        winner = 16384 if results[16384]["decision_score"] > results[32768]["decision_score"] else 32768
        print(f"\n16k comprime {100 * rel:.1f}% de lo que comprime 32k (umbral del plan: 81%)")
        print(f"score 16k = {results[16384]['decision_score']:.0f} | score 32k = {results[32768]['decision_score']:.0f} -> gana {winner}")
        results["winner"] = winner
        results["ratio_16k_vs_32k"] = rel

    out = REPO_ROOT / "benchmarks" / "tokenizer_comparison.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado en {out}")


if __name__ == "__main__":
    main()
