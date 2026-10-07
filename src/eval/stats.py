"""Estadistica minima para la evaluacion (solo numpy/stdlib)."""

from __future__ import annotations

from math import comb

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Media e intervalo de confianza (percentil) por bootstrap. Devuelve (media, bajo, alto); (None,)*3 si no hay datos."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimador insesgado de pass@k con n muestras de las que c aciertan (Chen et al., 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
