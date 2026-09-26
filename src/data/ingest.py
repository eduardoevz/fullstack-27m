"""Ingesta del corpus Full Stack desde parquet de HuggingFace, reanudable.

Por cada shard fuente: descarga -> filtro en cascada -> dedup -> shard .jsonl.gz.
El parquet se lee por row-groups (RAM constante) y se borra al terminar. El
manifiesto y el indice de dedup se guardan tras cada shard, asi que se puede
interrumpir y relanzar sin repetir trabajo.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.data.dedup import Deduper
from src.data.filters import filter_file

SOURCES = [
    ("codeparrot/github-code-clean", 880),
    ("codeparrot/github-code", 1126),
]
URL = "https://huggingface.co/datasets/{repo}/resolve/main/data/train-{i:05d}-of-{n:05d}.parquet"
COLUMNS = ["code", "repo_name", "path", "language"]


def download(url: str, dest: Path, retries: int = 5) -> None:
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            return
        except requests.RequestException as e:
            print(f"  descarga fallida ({e}); reintento {attempt + 1}/{retries}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"no se pudo descargar {url}")


def process_shard(parquet_path: Path, out_path: Path, cfg, dedup: Deduper) -> Counter:
    stats: Counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as out:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=COLUMNS)
            for code, repo, path in zip(*(table.column(c).to_pylist() for c in ("code", "repo_name", "path"))):
                stats["seen"] += 1
                lang, fw, reason = filter_file(code, path, cfg.data)
                if reason:
                    stats[f"rej_{reason}"] += 1
                    continue
                if not dedup.add_if_new(code):
                    stats["rej_duplicate"] += 1
                    continue
                stats["kept"] += 1
                stats["kept_bytes"] += len(code.encode("utf-8", errors="ignore"))
                stats[f"lang_{lang}"] += 1
                stats[f"fw_{fw}"] += 1
                out.write(json.dumps({"code": code, "lang": lang, "framework": fw,
                                      "path": path, "repo": repo}, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--target-gb", type=float, default=2.75, help="detener al acumular este volumen de codigo")
    ap.add_argument("--max-shards", type=int, default=None, help="limite de shards fuente a procesar en esta corrida")
    ap.add_argument("--source", type=int, default=0, help="indice en SOURCES")
    ap.add_argument("--dry-run", action="store_true", help="procesa 1 shard a un directorio temporal sin tocar el manifiesto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure()
    raw = Path(cfg.paths.raw_dir)
    repo, n_shards = SOURCES[args.source]

    if args.dry_run:
        raw = raw / "_dryrun"
        raw.mkdir(exist_ok=True)
        args.max_shards = 1

    manifest_path = raw / "manifest.json"
    dedup_path = raw / "dedup.npz"
    legacy_path = raw / "dedup.pkl"  # formato antiguo; se migra solo al primer guardado
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": repo, "done": [], "totals": {}}
    if manifest["source"] != repo:
        raise SystemExit(f"el manifiesto es de {manifest['source']}; usa otro --source o borra {raw}")
    if dedup_path.exists():
        dedup = Deduper.load(dedup_path)
    elif legacy_path.exists():
        dedup = Deduper.load(legacy_path)
    else:
        dedup = Deduper(cfg.data.dedup_threshold)
    if len(dedup) != manifest["totals"].get("kept", 0):
        raise SystemExit(f"indice de dedup ({len(dedup)}) no coincide con el manifiesto "
                         f"({manifest['totals'].get('kept', 0)}); no reanudo con estado inconsistente")
    totals = Counter(manifest["totals"])
    target = int(args.target_gb * 1e9)

    done = 0
    for i in range(n_shards):
        if totals["kept_bytes"] >= target:
            print(f"objetivo alcanzado: {totals['kept_bytes'] / 1e9:.2f} GB", flush=True)
            break
        if args.max_shards is not None and done >= args.max_shards:
            break
        if i in manifest["done"]:
            continue

        t0 = time.time()
        tmp = raw / f"_download-{i:05d}.parquet"
        print(f"[{i + 1}/{n_shards}] descargando...", flush=True)
        download(URL.format(repo=repo, i=i, n=n_shards), tmp)
        stats = process_shard(tmp, raw / f"shard-{i:05d}.jsonl.gz", cfg, dedup)
        tmp.unlink()

        totals.update(stats)
        manifest.update(done=manifest["done"] + [i], totals=dict(totals))
        dedup.save(dedup_path)
        gc.collect()
        manifest_path.write_text(json.dumps(manifest, indent=1))
        done += 1
        print(f"  visto={stats['seen']} conservado={stats['kept']} "
              f"({stats['kept_bytes'] / 1e6:.1f} MB) | acumulado {totals['kept_bytes'] / 1e9:.3f} GB | "
              f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
