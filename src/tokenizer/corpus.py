"""Lectura en streaming del corpus para el tokenizer.

- Hold-out por **repo** (crc32 estable, ~1 %): ningun archivo de esos repos entra
  al entrenamiento del tokenizer, asi que la compresion medida no es optimista.
  La misma particion debe reutilizarse para val.bin en la Fase 3.
- Muestreo del entrenamiento: cada documento se toma con probabilidad
  constante p = sample_bytes / bytes_totales. Da una muestra aleatoria repartida
  por todos los shards y con la misma mezcla de lenguajes que el corpus.
"""

from __future__ import annotations

import gzip
import json
import random
import zlib
from pathlib import Path
from typing import Iterator

HOLDOUT_MOD = 100  # 1 % de los repos


def is_holdout(repo: str) -> bool:
    return zlib.crc32(repo.encode("utf-8")) % HOLDOUT_MOD == 0


def iter_docs(raw_dir: str | Path) -> Iterator[dict]:
    for shard in sorted(Path(raw_dir).glob("shard-*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8", newline="") as f:
            for line in f:
                yield json.loads(line)


def iter_holdout(raw_dir: str | Path, max_docs: int | None = None) -> Iterator[dict]:
    n = 0
    for doc in iter_docs(raw_dir):
        if is_holdout(doc["repo"]):
            yield doc
            n += 1
            if max_docs is not None and n >= max_docs:
                return


def iter_train_texts(raw_dir: str | Path, sample_mb: float, seed: int = 0) -> Iterator[str]:
    manifest = json.loads((Path(raw_dir) / "manifest.json").read_text())
    p = min(1.0, sample_mb * 1e6 / manifest["totals"]["kept_bytes"] / (1 - 1 / HOLDOUT_MOD))
    rng = random.Random(seed)
    for doc in iter_docs(raw_dir):
        if not is_holdout(doc["repo"]) and rng.random() < p:
            yield doc["code"]
