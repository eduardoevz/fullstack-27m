"""Fase 11: cortes alineados a token para el diagnostico de frontera."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.fim_boundary import token_aligned_cases

NL = {20, 21}


def doc_ids(n_lines=30, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_lines):
        out += [int(t) for t in rng.integers(30, 90, size=int(rng.integers(2, 6)))] + [20 + int(rng.integers(0, 2))]
    return out


def test_cortes_alineados_reconstruyen_el_documento_y_caen_tras_un_salto():
    rng = np.random.default_rng(0)
    for i in range(30):
        ids = doc_ids(seed=i)
        k = 1 + i % 3
        pre, mid, suf = token_aligned_cases(ids, NL, k, rng)
        assert pre + mid + suf == ids
        assert pre[-1] in NL and mid[-1] in NL
        assert sum(t in NL for t in mid) == k
        assert sum(t in NL for t in pre) >= 3 and sum(t in NL for t in suf) >= 2


def test_documento_corto_devuelve_none():
    assert token_aligned_cases([30, 20, 31, 20], NL, 2, np.random.default_rng(0)) is None
