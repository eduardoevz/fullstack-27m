"""Fase 10: el prompt FIM de inferencia coincide con el formato de entrenamiento."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.sample import frame_fim_prompt, frame_prompt
from src.tokenizer.extend import add_fim_tokens, fim_ids
from test_encode import TEXTS, tok  # noqa: F401  (fixture)


def test_estructura_del_prompt(tok):
    t = add_fim_tokens(tok)
    f = fim_ids(t)
    ids = frame_fim_prompt(t, "const a = ", ";\nexport {a};\n", "tsx")
    pre = t.encode("const a = ").ids
    suf = t.encode(";\nexport {a};\n").ids
    assert ids == [t.token_to_id("<|file|>"), t.token_to_id("<|lang_ts|>"), f.prefix, *pre, f.suffix, *suf, f.middle]


def test_sufijo_vacio_y_prefijo_igual_que_el_normal(tok):
    t = add_fim_tokens(tok)
    f = fim_ids(t)
    ids = frame_fim_prompt(t, TEXTS[0], "", "tsx")
    assert ids[:2] == frame_prompt(t, TEXTS[0], "tsx")[:2]
    assert ids[-2:] == [f.suffix, f.middle]
