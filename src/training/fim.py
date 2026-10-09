"""Fill-in-the-Middle (Fase 10): formato PSM y muestreador de batches.

Un documento enmarcado  <|file|> <|lang|> CUERPO <|endoftext|>  pasa a
    <|file|> <|lang|> <|fim_prefix|> PREFIJO <|fim_suffix|> SUFIJO <|fim_middle|> MEDIO <|endoftext|>
con PREFIJO + MEDIO + SUFIJO = CUERPO (cortes al azar a nivel de token: puede partir un token BPE;
limitacion conocida). Una fila FIM contiene un documento transformado y se rellena con los
documentos siguientes del flujo, asi que no hay relleno ni cambia la perdida.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.training.data import BatchSampler

HEADER = 2          # <|file|> <|lang|>
FIM_TOKENS = 3      # prefix, suffix, middle
TAIL = 1            # <|endoftext|>


@dataclass(frozen=True)
class FimSpec:
    file: int
    eot: int
    prefix: int
    suffix: int
    middle: int


def doc_bounds(data: np.ndarray, file_id: int, eot_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices de inicio (<|file|>) y fin (<|endoftext|>, inclusive) de cada documento completo."""
    starts = np.flatnonzero(data == file_id)
    eots = np.flatnonzero(data == eot_id)
    k = np.searchsorted(eots, starts)
    ok = k < len(eots)
    return starts[ok].astype(np.int64), eots[k[ok]].astype(np.int64)


def fim_transform(doc: np.ndarray, rng: np.random.Generator, spec: FimSpec, max_len: int,
                  newline_ids: np.ndarray | None = None) -> np.ndarray:
    """Documento enmarcado -> PSM. Si no cabe en `max_len`, usa un tramo contiguo al azar del cuerpo.

    Con `newline_ids`, los dos cortes caen justo despues de un token que contiene un salto de linea (o en los
    extremos del cuerpo): el medio son lineas completas, como al autocompletar en un editor. Si no hay dos
    candidatos distintos, cae al corte aleatorio."""
    body = doc[HEADER:-TAIL]
    if len(body) < 2:
        return doc.copy()
    budget = max_len - HEADER - FIM_TOKENS - TAIL
    cut_ends = True
    if len(body) > budget:
        s = int(rng.integers(0, len(body) - budget + 1))
        body = body[s:s + budget]
        cut_ends = False                    # los extremos del tramo no son limites de linea reales
    cand = None
    if newline_ids is not None:
        cand = np.flatnonzero(np.isin(body, newline_ids)) + 1
        if cut_ends:
            cand = np.unique(np.concatenate([[0], cand, [len(body)]]))
        if len(cand) < 2:
            cand = None
    if cand is not None:
        a, b = np.sort(rng.choice(cand, size=2, replace=False))
    else:
        a, b = np.sort(rng.choice(len(body) + 1, size=2, replace=False))
    pre, mid, suf = body[:a], body[a:b], body[b:]
    return np.concatenate([
        doc[:HEADER], [spec.prefix], pre, [spec.suffix], suf, [spec.middle], mid, [spec.eot],
    ]).astype(np.uint16)


def split_psm(row: np.ndarray, spec: FimSpec):
    """Deshace PSM: devuelve (prefijo, sufijo, medio) de un documento FIM (hasta su <|endoftext|>)."""
    row = np.asarray(row)
    p = int(np.flatnonzero(row == spec.prefix)[0])
    s = int(np.flatnonzero(row == spec.suffix)[0])
    m = int(np.flatnonzero(row == spec.middle)[0])
    e = int(np.flatnonzero(row[m:] == spec.eot)[0]) + m
    return row[p + 1:s], row[s + 1:m], row[m + 1:e]


def fim_row(data, starts, ends, offset: int, row_len: int, rng: np.random.Generator, spec: FimSpec,
            newline_ids: np.ndarray | None = None) -> np.ndarray:
    """Fila de `row_len` tokens: el documento que contiene `offset` en PSM + el flujo que le sigue."""
    k = max(int(np.searchsorted(starts, offset, side="right")) - 1, 0)
    doc = np.asarray(data[starts[k]:ends[k] + 1])
    out = [fim_transform(doc, rng, spec, row_len, newline_ids)]
    have = len(out[0])
    pos = int(ends[k]) + 1
    while have < row_len:
        if pos >= len(data):
            pos = 0
        take = np.asarray(data[pos:pos + row_len - have])
        out.append(take)
        have += len(take)
        pos += len(take)
    return np.concatenate(out).astype(np.uint16)


class FimBatchSampler(BatchSampler):
    """Como BatchSampler, pero una fraccion `fim_rate` de las filas es FIM. Con 0 es identico al plano."""

    def __init__(self, bin_path: str | Path, block_size: int, batch_size: int, seed: int,
                 spec: FimSpec, fim_rate: float = 0.5, line_rate: float = 0.0, newline_ids=None):
        super().__init__(bin_path, block_size, batch_size, seed)
        self.spec, self.fim_rate, self.line_rate = spec, fim_rate, line_rate
        self.newline_ids = None if newline_ids is None else np.asarray(sorted(newline_ids), dtype=np.uint16)
        if line_rate > 0 and self.newline_ids is None:
            raise ValueError("line_rate > 0 requiere newline_ids")
        self.starts, self.ends = doc_bounds(self.data, spec.file, spec.eot)
        if fim_rate > 0 and not len(self.starts):
            raise ValueError(f"{bin_path}: no hay documentos enmarcados para FIM")

    def _rows(self) -> np.ndarray:
        n = self.block_size + 1
        offsets = self.rng.integers(0, len(self.data) - self.block_size - 1, size=self.batch_size)
        use_fim = self.rng.random(self.batch_size) < self.fim_rate if self.fim_rate > 0 else np.zeros(self.batch_size, bool)
        use_line = (self.rng.random(self.batch_size) < self.line_rate
                    if self.line_rate > 0 and self.fim_rate > 0 else np.zeros(self.batch_size, bool))
        rows = [fim_row(self.data, self.starts, self.ends, int(o), n, self.rng, self.spec,
                        self.newline_ids if ln else None) if f else self.data[o:o + n]
                for o, f, ln in zip(offsets, use_fim, use_line)]
        return np.stack(rows).astype(np.int64)

    def get_batch(self):
        import torch
        t = torch.from_numpy(self._rows())
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()
