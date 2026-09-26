"""Deduplicacion casi-exacta con MinHash + LSH sobre 5-gramas de tokens.

Implementacion propia con numpy (sin `datasketch`). Cada archivo aceptado se
guarda como una firma de 128 x uint32 (512 B). Las bandas LSH proponen
candidatos y la firma confirma con el Jaccard estimado >= umbral.

Memoria: el indice LSH son arrays numpy ordenados (12 B por banda y archivo)
en lugar de dicts de Python (~250 B por entrada). A ~400k archivos pasa de
2.3 GB a ~0.4 GB. Los archivos nuevos van a un buffer pequeno que se funde en
los arrays ordenados en `flush()` (se llama al guardar).
"""

from __future__ import annotations

import pickle
import re
import zlib
from pathlib import Path

import numpy as np

NUM_PERM = 128
BANDS = 16                      # 16 bandas x 8 filas: punto de corte LSH ~0.71
ROWS = NUM_PERM // BANDS
PRIME = 4294967311              # primo > 2**32
TOKEN = re.compile(r"\w+|[^\w\s]")
SHINGLE_SIZE = 5
_MIX = np.random.default_rng(99).integers(1, 2**63, size=ROWS, dtype=np.uint64) | np.uint64(1)


def shingles(text: str) -> set[int]:
    toks = TOKEN.findall(text)
    if len(toks) <= SHINGLE_SIZE:
        return {zlib.crc32(" ".join(toks).encode("utf-8", errors="ignore"))}
    return {
        zlib.crc32(" ".join(toks[i:i + SHINGLE_SIZE]).encode("utf-8", errors="ignore"))
        for i in range(len(toks) - SHINGLE_SIZE + 1)
    }


def band_keys(sigs: np.ndarray) -> np.ndarray:
    """(N, NUM_PERM) uint32 -> (N, BANDS) uint64: una clave hash por banda."""
    s = sigs.reshape(len(sigs), BANDS, ROWS).astype(np.uint64)
    with np.errstate(over="ignore"):
        return (s * _MIX).sum(axis=2, dtype=np.uint64)


class Deduper:
    def __init__(self, threshold: float = 0.8, seed: int = 1234):
        self.threshold = threshold
        self.seed = seed
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, PRIME, size=NUM_PERM, dtype=np.uint64)
        self._b = rng.integers(0, PRIME, size=NUM_PERM, dtype=np.uint64)
        self._sigs = np.empty((1024, NUM_PERM), dtype=np.uint32)
        self._n = 0
        self._keys = [np.empty(0, dtype=np.uint64) for _ in range(BANDS)]   # ordenadas
        self._ids = [np.empty(0, dtype=np.uint32) for _ in range(BANDS)]
        self._pending: list[dict[int, list[int]]] = [{} for _ in range(BANDS)]
        self._merged = 0                                                   # ids ya fundidos

    def __len__(self) -> int:
        return self._n

    def _signature(self, text: str) -> np.ndarray:
        h = np.fromiter(shingles(text), dtype=np.uint64)
        # (a*h + b) mod p; a,h < 2**32 asi que a*h cabe en uint64 sin desbordar
        vals = (self._a[:, None] * h[None, :] + self._b[:, None]) % PRIME
        return vals.min(axis=1).astype(np.uint32)

    def _candidates(self, keys: np.ndarray) -> set[int]:
        found: set[int] = set()
        for b in range(BANDS):
            k = keys[b]
            lo = np.searchsorted(self._keys[b], k, side="left")
            hi = np.searchsorted(self._keys[b], k, side="right")
            if hi > lo:
                found.update(self._ids[b][lo:hi].tolist())
            found.update(self._pending[b].get(int(k), ()))
        return found

    def add_if_new(self, text: str) -> bool:
        """True si el texto es nuevo (y lo registra); False si es duplicado."""
        sig = self._signature(text)
        keys = band_keys(sig[None, :])[0]
        cands = self._candidates(keys)
        if cands:
            idx = np.fromiter(cands, dtype=np.int64)
            if (np.mean(self._sigs[idx] == sig, axis=1) >= self.threshold).any():
                return False
        if self._n == len(self._sigs):
            self._sigs = np.concatenate([self._sigs, np.empty_like(self._sigs)])
        self._sigs[self._n] = sig
        for b in range(BANDS):
            self._pending[b].setdefault(int(keys[b]), []).append(self._n)
        self._n += 1
        return True

    def flush(self) -> None:
        """Funde el buffer en los arrays ordenados."""
        if self._merged == self._n:
            return
        new_keys = band_keys(self._sigs[self._merged:self._n])
        new_ids = np.arange(self._merged, self._n, dtype=np.uint32)
        for b in range(BANDS):
            keys = np.concatenate([self._keys[b], new_keys[:, b]])
            ids = np.concatenate([self._ids[b], new_ids])
            order = np.argsort(keys, kind="stable")
            self._keys[b], self._ids[b] = keys[order], ids[order]
            self._pending[b] = {}
        self._merged = self._n

    def save(self, path: str | Path) -> None:
        """Guarda solo las firmas (el indice se reconstruye al cargar)."""
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, sigs=self._sigs[:self._n], threshold=self.threshold, seed=self.seed)
        tmp.replace(path)
        self.flush()

    @classmethod
    def _from_sigs(cls, sigs: np.ndarray, threshold: float, seed: int) -> "Deduper":
        d = cls(threshold=threshold, seed=seed)
        d._sigs = np.ascontiguousarray(sigs)
        d._n = len(sigs)
        if d._n == 0:
            d._sigs = np.empty((1024, NUM_PERM), dtype=np.uint32)
        d._merged = 0
        d.flush()
        return d

    @classmethod
    def load(cls, path: str | Path) -> "Deduper":
        path = Path(path)
        if path.suffix == ".pkl":  # formato antiguo: solo se aprovechan las firmas
            with open(path, "rb") as f:
                state = pickle.load(f)
            sigs = np.array(state["sigs"], dtype=np.uint32).reshape(-1, NUM_PERM)
            return cls._from_sigs(sigs, state["threshold"], state["seed"])
        with np.load(path) as z:
            return cls._from_sigs(z["sigs"], float(z["threshold"]), int(z["seed"]))
