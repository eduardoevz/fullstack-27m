"""Ingesta de una fuente moderna (StarCoderData) enfocada en Next.js, FastAPI, NestJS y TSX.

Fase 9. Por cada shard: descarga (con token) -> filtro en cascada -> solo los frameworks
buscados -> limpieza (dts, plantillas, Python 2, licencias) -> tope por framework -> dedup
contra el corpus ya ingerido y contra lo nuevo -> shard .jsonl.gz con el mismo formato que
`raw/`. Reanudable: manifiesto y dedup se guardan tras cada shard. Escribe en un directorio
propio (`raw_modern/`) para no mezclar fuentes ni romper el manifiesto de `raw/`.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import load_config
from src.data.dedup import Deduper
from src.data.filters import clean_document, filter_file, quality_reject
from src.data.hf_auth import load_hf_token

DATASET = "bigcode/starcoderdata"
# Directorios del dataset (uno por lenguaje) que se recorren por defecto.
DEFAULT_DIRS = ("typescript", "python")
COLUMNS = ["content", "max_stars_repo_name", "max_stars_repo_path"]
# Frameworks que se conservan. React solo cuenta en .tsx (objetivo de mezcla: tsx >= 15 %).
WANTED = {"nextjs", "fastapi", "nestjs", "react"}
# Topes en bytes de codigo por framework (los escasos -nextjs, fastapi- no tienen tope).
DEFAULT_CAPS = {"react": 400_000_000, "nestjs": 250_000_000}


def select_document(code, path, repo, cfg, dedup, caps, used, stats):
    """Devuelve el registro a guardar, o None (anotando el motivo en `stats`)."""
    stats["seen"] += 1
    lang, fw, reason = filter_file(code, path, cfg)
    if reason:
        stats[f"rej_{reason}"] += 1
        return None
    if fw not in WANTED or (fw == "react" and lang != "tsx"):
        stats["rej_unwanted"] += 1
        return None
    cleaned, reason = clean_document(code, path, lang)
    if reason:
        stats[f"rej_{reason}"] += 1
        return None
    reason = quality_reject(cleaned, cfg)
    if reason:
        stats[f"rej_{reason}"] += 1
        return None
    size = len(cleaned.encode("utf-8", errors="ignore"))
    if fw in caps and used[fw] >= caps[fw]:
        stats["rej_cap"] += 1
        return None
    if not dedup.add_if_new(cleaned):
        stats["rej_duplicate"] += 1
        return None
    used[fw] += size
    stats["kept"] += 1
    stats["kept_bytes"] += size
    stats[f"lang_{lang}"] += 1
    stats[f"fw_{fw}"] += 1
    stats[f"bytes_{fw}"] += size
    return {"code": cleaned, "lang": lang, "framework": fw, "path": path, "repo": repo,
            "source": "starcoderdata"}


def list_shards(api, dirs, token):
    out = []
    for d in dirs:
        names = sorted(e.path for e in api.list_repo_tree(DATASET, path_in_repo=d, repo_type="dataset", token=token)
                       if e.path.endswith(".parquet"))
        out += names
    return out


def fetch(name, dest_dir, token, retries=5):
    from huggingface_hub import hf_hub_download
    for attempt in range(retries):
        try:
            return Path(hf_hub_download(DATASET, name, repo_type="dataset", token=token, local_dir=dest_dir))
        except Exception as e:  # red/HF: reintentar con espera
            print(f"  descarga fallida ({type(e).__name__}); reintento {attempt + 1}/{retries}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"no se pudo descargar {name}")


def process_shard(parquet_path, out_path, cfg, dedup, caps, used):
    import pyarrow.parquet as pq
    stats: Counter = Counter()
    pf = pq.ParquetFile(parquet_path)
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as out:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=COLUMNS)
            cols = [t.column(c).to_pylist() for c in COLUMNS]
            for code, repo, path in zip(*cols):
                if not code or not path:
                    continue
                rec = select_document(code, path, repo, cfg, dedup, caps, used, stats)
                if rec:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dirs", default=",".join(DEFAULT_DIRS), help="directorios de lenguaje de StarCoderData")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="1 shard a un directorio temporal, sin tocar el manifiesto real")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure()
    base_raw = Path(cfg.paths.raw_dir)
    raw = base_raw.parent / "raw_modern"
    if args.dry_run:
        raw = raw / "_dryrun"
        args.max_shards = 1
    raw.mkdir(parents=True, exist_ok=True)
    token = load_hf_token()
    if not token:
        raise SystemExit("falta HF_TOKEN (variable de entorno o .env)")

    manifest_path, dedup_path = raw / "manifest.json", raw / "dedup.npz"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else \
        {"source": DATASET, "done": [], "totals": {}, "base_n": None}
    if dedup_path.exists():
        dedup = Deduper.load(dedup_path)
    else:  # empieza con el corpus ya ingerido, para no repetir lo que ya tenemos
        dedup = Deduper.load(base_raw / "dedup.npz")
        manifest["base_n"] = len(dedup)
    totals = Counter(manifest["totals"])
    if len(dedup) != manifest["base_n"] + totals["kept"]:
        raise SystemExit("indice de dedup inconsistente con el manifiesto; no reanudo")
    used = Counter({k[len("bytes_"):]: v for k, v in totals.items() if k.startswith("bytes_")})

    from huggingface_hub import HfApi
    shards = list_shards(HfApi(), args.dirs.split(","), token)
    tmp_dir = raw / "_download"
    done = 0
    for n, name in enumerate(shards, 1):
        if name in manifest["done"]:
            continue
        if args.max_shards is not None and done >= args.max_shards:
            break
        t0 = time.time()
        print(f"[{n}/{len(shards)}] {name}", flush=True)
        path = fetch(name, tmp_dir, token)
        out_name = name.replace("/", "-").replace(".parquet", ".jsonl.gz")
        stats = process_shard(path, raw / out_name, cfg.data, dedup, DEFAULT_CAPS, used)
        path.unlink()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        totals.update(stats)
        manifest.update(done=manifest["done"] + [name], totals=dict(totals))
        dedup.save(dedup_path)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        gc.collect()
        done += 1
        fws = {k[3:]: v for k, v in stats.items() if k.startswith("fw_")}
        print(f"  visto={stats['seen']} conservado={stats['kept']} ({stats['kept_bytes'] / 1e6:.1f} MB) "
              f"docs por framework={fws} | acumulado {totals['kept_bytes'] / 1e9:.3f} GB | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
