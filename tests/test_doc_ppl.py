"""Fase 9, paso 5: puntuacion de documentos con v1 y corte de la cola peor por dominio."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.eval.doc_ppl import doc_mean_losses, pad_batch, tail_thresholds, worst_tail_mask


@pytest.fixture(scope="module")
def model():
    from src.model.gpt import GPT
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64, block_size=64,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
    return GPT(cfg).eval()


def test_pad_batch_trunca_y_rellena():
    rows, lens = pad_batch([[1, 2, 3, 4, 5, 6], [7, 8]], max_len=4, pad_id=0)
    assert rows.tolist() == [[1, 2, 3, 4], [7, 8, 0, 0]]
    assert lens.tolist() == [4, 2]


def test_puntuacion_no_depende_del_relleno_ni_del_lote(model):
    docs = [list(range(5, 25)), list(range(30, 38)), list(range(40, 60))]
    a = doc_mean_losses(model, docs, max_len=16, batch=3, pad_id=0, skip_header=2)
    b = doc_mean_losses(model, [docs[1]], max_len=16, batch=1, pad_id=0, skip_header=2)
    c = doc_mean_losses(model, [docs[1]], max_len=16, batch=1, pad_id=7, skip_header=2)
    assert a[1] == pytest.approx(b[0], abs=1e-5) == pytest.approx(c[0], abs=1e-5)
    assert np.isfinite(a).all() and (a > 0).all()


def test_documentos_demasiado_cortos_dan_nan(model):
    out = doc_mean_losses(model, [[5, 6]], max_len=16, batch=1, pad_id=0, skip_header=2)
    assert np.isnan(out[0])          # tras quitar la cabecera no queda ningun token objetivo


def test_cola_peor_por_grupo_no_mezcla_dominios():
    scores = np.array([1.0, 1.1, 1.2, 1.3, 9.0] + [3.0, 3.1, 3.2, 3.3, 3.4] * 3)
    groups = np.array(["a"] * 5 + ["b"] * 15)
    th = tail_thresholds(scores, groups, pct=20)
    mask = worst_tail_mask(scores, groups, th)
    assert mask[4]                                     # el 9.0 es lo peor de "a"
    assert not mask[:4].any()
    assert mask[5:].sum() == 3                         # 20 % de "b": solo sus peores; "a" no arrastra a "b"
    assert 3.0 not in scores[mask]


def test_nan_no_se_descarta():
    scores = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    groups = np.array(["a"] * 5)
    th = tail_thresholds(scores, groups, pct=20)
    assert not worst_tail_mask(scores, groups, th)[1]
