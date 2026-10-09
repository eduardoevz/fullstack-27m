"""Fase 10: los tokens FIM se anaden al final del vocabulario sin mover ningun id existente."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tokenizer.extend import add_fim_tokens, fim_ids
from src.tokenizer.pretokenizer import FIM_TOKENS
from test_encode import TEXTS, tok  # noqa: F401  (fixture)

SAMPLE = TEXTS[0] + TEXTS[1] + "<|file|><|lang_ts|>x<|endoftext|>"


def test_ids_nuevos_al_final_y_los_viejos_no_cambian(tok):
    n = tok.get_vocab_size()
    new = add_fim_tokens(tok)
    assert new.get_vocab_size() == n + len(FIM_TOKENS)
    assert [new.token_to_id(t) for t in FIM_TOKENS] == list(range(n, n + len(FIM_TOKENS)))
    for tok_str, i in tok.get_vocab().items():
        assert new.token_to_id(tok_str) == i


def test_la_codificacion_de_texto_normal_no_cambia(tok):
    new = add_fim_tokens(tok)
    assert new.encode(SAMPLE, add_special_tokens=False).ids == tok.encode(SAMPLE, add_special_tokens=False).ids


def test_los_tokens_fim_son_atomicos_y_se_decodifican(tok):
    new = add_fim_tokens(tok)
    ids = new.encode("a<|fim_prefix|>b<|fim_suffix|>c<|fim_middle|>d", add_special_tokens=False).ids
    assert [i for i in ids if i >= tok.get_vocab_size()] == [new.token_to_id(t) for t in FIM_TOKENS]
    assert new.decode(ids, skip_special_tokens=False) == "a<|fim_prefix|>b<|fim_suffix|>c<|fim_middle|>d"


def test_no_modifica_el_original_y_es_idempotente(tok):
    n = tok.get_vocab_size()
    add_fim_tokens(tok)
    assert tok.get_vocab_size() == n
    twice = add_fim_tokens(add_fim_tokens(tok))
    assert twice.get_vocab_size() == n + len(FIM_TOKENS)


def test_fim_ids(tok):
    new = add_fim_tokens(tok)
    f = fim_ids(new)
    assert (f.prefix, f.suffix, f.middle) == tuple(new.token_to_id(t) for t in FIM_TOKENS)
    with pytest.raises(ValueError):
        fim_ids(tok)           # sin los tokens, error claro
