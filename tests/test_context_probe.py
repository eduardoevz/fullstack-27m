"""Fase 12 (medicion previa): reglas fijadas antes de medir si el contexto de 512 limita y si un dominio es flojo."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.context_probe import classify_context, classify_domain, share_beyond, target_window


def test_clasifica_limita_cuando_el_contexto_largo_aun_ayuda_y_hay_mucho_texto_largo():
    losses = {64: 1.60, 128: 1.50, 256: 1.40, 384: 1.36, 512: 1.34}
    assert classify_context(losses, share_long=0.40) == "limita"


def test_clasifica_no_limita_si_el_ultimo_tramo_apenas_ayuda():
    losses = {64: 1.60, 128: 1.50, 256: 1.40, 384: 1.396, 512: 1.395}
    assert classify_context(losses, share_long=0.40) == "no limita"


def test_clasifica_dudoso_en_el_medio_o_con_poco_texto_largo():
    losses = {64: 1.60, 128: 1.50, 256: 1.40, 384: 1.36, 512: 1.34}
    assert classify_context(losses, share_long=0.10) == "dudoso"
    assert classify_context({64: 1.5, 128: 1.45, 256: 1.40, 384: 1.39, 512: 1.38}, share_long=0.40) == "dudoso"


def test_dominio_flojo_si_supera_el_10_por_ciento_a_la_mezcla():
    assert classify_domain(1.80, 1.50) == "flojo"
    assert classify_domain(1.60, 1.50) == "ok"


def test_ventana_de_objetivo_termina_en_el_mismo_token_para_cualquier_contexto():
    data = np.arange(2000, dtype=np.uint16)
    for c in (64, 256, 512):
        row = target_window(data, start=100, context=c)
        assert row[-1] == 100 + 512 and len(row) == c + 1       # entrada de c tokens + 1 de objetivo, mismo ultimo token


def test_share_beyond_cuenta_tokens_despues_de_la_posicion_del_documento():
    assert share_beyond([100, 600, 1100], 512) == (88 + 588) / 1800 + 0.0
