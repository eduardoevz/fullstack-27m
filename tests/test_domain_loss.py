"""Fase 9, paso 7: perdida de un modelo sobre un flujo de tokens, en ventanas consecutivas."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.eval.domain_loss import stream_windows, stream_loss


@pytest.fixture(scope="module")
def model():
    from src.model.gpt import GPT
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64, block_size=64,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
    return GPT(cfg).eval()


def test_ventanas_consecutivas_sin_solapar_ni_sobrar():
    data = np.arange(100, dtype=np.uint16)
    w = stream_windows(data, block=16)
    assert w.shape == (6, 17)
    assert w[0].tolist() == list(range(17)) and w[1][0] == 16      # cada ventana empieza donde acaba el bloque anterior


def test_max_windows_recorta_de_forma_determinista():
    data = (np.arange(2000) % 50).astype(np.uint16)
    a = stream_windows(data, block=16, max_windows=10, seed=1)
    b = stream_windows(data, block=16, max_windows=10, seed=1)
    assert a.shape == (10, 17) and np.array_equal(a, b)


def test_stream_loss_es_finita_y_reproducible(model):
    data = (np.arange(3000) % 60 + 2).astype(np.uint16)
    r1 = stream_loss(model, data, block=32, batch=4)
    r2 = stream_loss(model, data, block=32, batch=7)
    assert r1["loss"] == pytest.approx(r2["loss"], abs=1e-4) and r1["loss"] > 0
    assert r1["tokens"] == r1["windows"] * 32
