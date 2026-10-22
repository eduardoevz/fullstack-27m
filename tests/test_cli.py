"""Fase 13: la demo completa en el cursor (<CURSOR>) con el formato FIM y evita los marcadores en el modo normal."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import CURSOR_MARK, complete, split_cursor
from src.config import ModelConfig
from src.model.gpt import GPT
from src.sample import fim_banned_ids
from src.tokenizer.extend import add_fim_tokens
from test_encode import tok  # noqa: F401  (fixture)


def tiny_for(t, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=t.get_vocab_size(), n_layer=2, n_head=4, d_model=64, d_ff=128, block_size=64,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
    m = GPT(cfg).eval()
    for p in m.parameters():
        p.data.mul_(8)
    return m


def test_split_cursor_separa_prefijo_y_sufijo():
    assert split_cursor(f"def f(a):\n    {CURSOR_MARK}\n    return r\n") == ("def f(a):\n    ", "\n    return r\n")


def test_split_cursor_sin_marca_devuelve_none():
    assert split_cursor("def f(a):\n    return a\n") is None


def test_split_cursor_usa_el_primero_y_admite_sufijo_vacio():
    assert split_cursor(f"a{CURSOR_MARK}") == ("a", "")
    pre, suf = split_cursor(f"a{CURSOR_MARK}b{CURSOR_MARK}c")
    assert pre == "a" and suf == f"b{CURSOR_MARK}c"


def test_complete_respeta_el_presupuesto_en_modo_fim(tok):
    t = add_fim_tokens(tok)
    out = complete(tiny_for(t), t, f"const a = {CURSOR_MARK};\n", "ts", 12, 0.0, 1.0)
    assert isinstance(out, str)
    assert len(t.encode(out).ids) <= 12 + 4        # re-tokenizar puede partir distinto; margen pequeno


def test_complete_en_modo_normal_nunca_emite_marcadores_fim(tok):
    t = add_fim_tokens(tok)
    model = tiny_for(t)
    banned = set(fim_banned_ids(t))
    ids = []
    complete(model, t, "const a = 1;\n", "ts", 30, 1.0, 1.0, on_token=ids.append)
    assert ids and not set(ids) & banned
