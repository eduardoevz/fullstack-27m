"""Estadisticas del corpus: lenguajes, frameworks, rechazos y tamanos.

Usa el manifiesto para los contadores globales y, opcionalmente, muestrea
archivos del corpus para inspeccion manual (--sample N).
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", type=int, default=0, help="imprime N archivos aleatorios (cabecera) para inspeccion")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw = Path(cfg.paths.raw_dir)
    manifest = json.loads((raw / "manifest.json").read_text())
    t = manifest["totals"]

    seen, kept = t.get("seen", 0), t.get("kept", 0)
    print(f"Fuente: {manifest['source']} | shards procesados: {len(manifest['done'])}")
    print(f"Archivos vistos: {seen:,} | conservados: {kept:,} ({100 * kept / max(seen, 1):.2f}%)")
    print(f"Codigo conservado: {t.get('kept_bytes', 0) / 1e9:.3f} GB\n")

    def block(title: str, prefix: str) -> None:
        rows = sorted(((k[len(prefix):], v) for k, v in t.items() if k.startswith(prefix)), key=lambda kv: -kv[1])
        print(title)
        total = sum(v for _, v in rows) or 1
        for name, v in rows:
            print(f"  {name:<14}{v:>9,}  {100 * v / total:5.1f}%")
        print()

    block("Por lenguaje", "lang_")
    block("Por framework/senal de dominio", "fw_")
    block("Rechazos por filtro", "rej_")

    if args.sample:
        files = sorted(raw.glob("shard-*.jsonl.gz"))
        rng = random.Random(0)
        for _ in range(args.sample):
            with gzip.open(rng.choice(files), "rt", encoding="utf-8") as f:
                docs = [json.loads(line) for line in f]
            d = rng.choice(docs)
            print("=" * 70)
            print(f"{d['repo']}/{d['path']}  [{d['lang']} | {d['framework']}]")
            print("\n".join(d["code"].splitlines()[:12]))


if __name__ == "__main__":
    main()
