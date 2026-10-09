"""Fase 10: cargar los pesos de v1 en un modelo con mas filas de embedding (tokens FIM)."""
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.model.gpt import GPT
from src.training.init_from import load_v1_weights

OLD = ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64, block_size=64,
                  rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
NEW = replace(OLD, vocab_size=67)


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    old = GPT(OLD).eval()
    p = tmp_path / "v1.pt"
    torch.save({"model": old.state_dict(), "step": 15000, "best_val": 1.47}, p)
    return old, p


def test_filas_viejas_identicas_y_nuevas_son_la_media(ckpt):
    old, p = ckpt
    new = GPT(NEW)
    info = load_v1_weights(new, p)
    w_old, w_new = old.tok_emb.weight, new.tok_emb.weight
    assert torch.equal(w_new[:64], w_old)
    assert torch.allclose(w_new[64:], w_old.mean(0, keepdim=True).expand(3, -1))
    assert info == {"step": 15000, "old_vocab": 64, "new_vocab": 67}
    assert new.lm_head.weight.data_ptr() == new.tok_emb.weight.data_ptr()       # sigue atado


def test_el_resto_de_pesos_se_copia_tal_cual(ckpt):
    old, p = ckpt
    new = GPT(NEW)
    load_v1_weights(new, p)
    for (k, a), (_, b) in zip(old.state_dict().items(), new.state_dict().items()):
        if k not in ("tok_emb.weight", "lm_head.weight"):
            assert torch.equal(a, b), k


def test_logits_de_los_ids_viejos_no_cambian(ckpt):
    old, p = ckpt
    new = GPT(NEW).eval()
    load_v1_weights(new, p)
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        lo, _ = old(x)
        ln, _ = new(x)
    assert ln.shape[-1] == 67
    assert torch.allclose(ln[..., :64], lo, atol=1e-5)


def test_mismo_tamano_funciona_y_arquitectura_distinta_falla(ckpt, tmp_path):
    old, p = ckpt
    same = GPT(OLD)
    load_v1_weights(same, p)
    assert torch.equal(same.tok_emb.weight, old.tok_emb.weight)
    with pytest.raises((ValueError, RuntimeError)):
        load_v1_weights(GPT(replace(OLD, d_model=64, d_ff=128)), p)
    with pytest.raises(ValueError):
        load_v1_weights(GPT(replace(OLD, vocab_size=60)), p)         # recortar el vocabulario no se admite
