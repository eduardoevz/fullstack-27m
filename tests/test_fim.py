"""Fase 10: formato FIM (PSM) y muestreador. Ids de juguete: file=1, eot=2, lang=3, FIM=100/101/102."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data import BatchSampler
from src.training.fim import FimBatchSampler, FimSpec, doc_bounds, fim_row, fim_transform, split_psm

SPEC = FimSpec(file=1, eot=2, prefix=100, suffix=101, middle=102)


def make_stream(n_docs=300, seed=0, lead_fragment=True):
    rng = np.random.default_rng(seed)
    parts = [np.array([50, 51, 52], dtype=np.uint16)] if lead_fragment else []
    docs = []
    for _ in range(n_docs):
        body = rng.integers(10, 99, size=int(rng.integers(5, 400)), dtype=np.uint16)
        docs.append(body)
        parts.append(np.concatenate([[1, 3], body, [2]]).astype(np.uint16))
    return np.concatenate(parts), docs


@pytest.fixture
def stream_file(tmp_path):
    data, docs = make_stream()
    p = tmp_path / "s.bin"
    data.tofile(p)
    return p, data, docs


def test_doc_bounds_ignora_fragmento_inicial():
    data, docs = make_stream(10)
    starts, ends = doc_bounds(data, SPEC.file, SPEC.eot)
    assert len(starts) == 10 and starts[0] == 3
    assert all(data[s] == 1 and data[e] == 2 for s, e in zip(starts, ends))
    assert [int(e - s + 1) for s, e in zip(starts, ends)] == [len(d) + 3 for d in docs]


def test_fim_transform_reconstruye_el_cuerpo():
    rng = np.random.default_rng(1)
    for body_len in (2, 3, 10, 200):
        body = (10 + np.arange(body_len) % 80).astype(np.uint16)
        doc = np.concatenate([[1, 3], body, [2]]).astype(np.uint16)
        out = fim_transform(doc, rng, SPEC, max_len=512)
        assert out[:3].tolist() == [1, 3, 100] and out[-1] == 2
        pre, suf, mid = split_psm(out, SPEC)
        assert np.array_equal(np.concatenate([pre, mid, suf]), body)
        assert len(mid) >= 1 and len(out) == body_len + 6
        for t in (100, 101, 102):
            assert int((out == t).sum()) == 1


def test_fim_transform_trunca_a_un_tramo_contiguo():
    rng = np.random.default_rng(2)
    body = np.arange(10, 90, dtype=np.uint16)
    body = np.tile(body, 20)                        # 1600 tokens
    doc = np.concatenate([[1, 3], body, [2]]).astype(np.uint16)
    out = fim_transform(doc, rng, SPEC, max_len=100)
    assert len(out) <= 100
    pre, suf, mid = split_psm(out, SPEC)
    seg = np.concatenate([pre, mid, suf])
    assert len(seg) == 94
    assert any(np.array_equal(seg, body[i:i + 94]) for i in range(len(body) - 93))


def test_fim_transform_cuerpo_demasiado_corto_devuelve_el_documento_igual():
    rng = np.random.default_rng(3)
    doc = np.array([1, 3, 20, 2], dtype=np.uint16)
    assert np.array_equal(fim_transform(doc, rng, SPEC, max_len=512), doc)


def test_fim_row_longitud_exacta_y_relleno_con_el_flujo(stream_file):
    _, data, _ = stream_file
    starts, ends = doc_bounds(data, SPEC.file, SPEC.eot)
    rng = np.random.default_rng(4)
    for o in (0, 1, 500, len(data) - 600):
        row = fim_row(data, starts, ends, o, 513, rng, SPEC)
        assert len(row) == 513 and row.dtype == np.uint16
        assert row[2] == SPEC.prefix
        e = int(np.flatnonzero(row == SPEC.eot)[0])           # fin del documento FIM
        k = int(np.searchsorted(starts, o, side="right")) - 1
        k = max(k, 0) + (1 if o > ends[max(k, 0)] and max(k, 0) + 1 < len(starts) else 0)
        nxt = int(ends[k]) + 1
        tail = row[e + 1:]
        assert np.array_equal(tail, data[nxt:nxt + len(tail)])  # lo que sigue es el flujo normal


def test_sampler_tasa_fim_determinismo_y_forma(stream_file):
    p, data, _ = stream_file
    a = FimBatchSampler(p, 128, 64, seed=7, spec=SPEC, fim_rate=0.5)
    x, y = a.get_batch()
    assert x.shape == (64, 128) and y.shape == (64, 128) and x.dtype == y.dtype
    assert (x[:, 1:] == y[:, :-1]).all()
    n_fim = fraction = 0
    for _ in range(20):
        x, _ = a.get_batch()
        n_fim += int((x[:, 2] == SPEC.prefix).sum())
    fraction = n_fim / (20 * 64)
    assert 0.40 < fraction < 0.60
    b = FimBatchSampler(p, 128, 64, seed=7, spec=SPEC, fim_rate=0.5)
    c = FimBatchSampler(p, 128, 64, seed=7, spec=SPEC, fim_rate=0.5)
    assert all(torch_equal(b.get_batch(), c.get_batch()) for _ in range(3))


def torch_equal(u, v):
    return bool((u[0] == v[0]).all() and (u[1] == v[1]).all())


def test_sampler_restaura_estado(stream_file):
    p, _, _ = stream_file
    a = FimBatchSampler(p, 128, 8, seed=1, spec=SPEC, fim_rate=0.5)
    a.get_batch()
    state = a.state_dict()
    nxt = a.get_batch()
    b = FimBatchSampler(p, 128, 8, seed=99, spec=SPEC, fim_rate=0.5)
    b.load_state_dict(state)
    assert torch_equal(nxt, b.get_batch())


def test_tasa_cero_equivale_exactamente_al_muestreador_plano(stream_file):
    p, _, _ = stream_file
    f = FimBatchSampler(p, 128, 8, seed=5, spec=SPEC, fim_rate=0.0)
    base = BatchSampler(p, 128, 8, seed=5)
    assert all(torch_equal(f.get_batch(), base.get_batch()) for _ in range(3))


# --- cortes alineados a limite de linea (ajuste tras la Fase 10) -----------------------------------
NL = np.array([20, 21], dtype=np.uint16)          # ids de juguete que contienen un salto de linea


def lined_doc(n_lines=30, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(n_lines):
        parts += [rng.integers(30, 90, size=int(rng.integers(2, 8)), dtype=np.uint16), [20 + int(rng.integers(0, 2))]]
    body = np.concatenate([np.asarray(p, dtype=np.uint16) for p in parts])
    return np.concatenate([[1, 3], body, [2]]).astype(np.uint16), body


def test_cortes_por_linea_caen_justo_despues_de_un_salto():
    rng = np.random.default_rng(0)
    for i in range(50):
        doc, body = lined_doc(seed=i)
        out = fim_transform(doc, rng, SPEC, max_len=512, newline_ids=NL)
        pre, suf, mid = split_psm(out, SPEC)
        assert np.array_equal(np.concatenate([pre, mid, suf]), body)
        assert len(pre) == 0 or pre[-1] in NL                  # el prefijo termina en salto (o esta vacio)
        assert mid[-1] in NL                                    # el medio son lineas completas
        assert len(mid) >= 1


def test_sin_candidatos_cae_a_cortes_aleatorios():
    rng = np.random.default_rng(1)
    body = (30 + np.arange(40) % 50).astype(np.uint16)          # sin ningun salto de linea
    doc = np.concatenate([[1, 3], body, [2]]).astype(np.uint16)
    out = fim_transform(doc, rng, SPEC, max_len=512, newline_ids=NL)
    pre, suf, mid = split_psm(out, SPEC)
    assert np.array_equal(np.concatenate([pre, mid, suf]), body) and len(mid) >= 1


def test_sampler_mezcla_cortes_de_linea_y_aleatorios(stream_file):
    p, _, _ = stream_file
    s = FimBatchSampler(p, 128, 64, seed=3, spec=SPEC, fim_rate=1.0, line_rate=0.5, newline_ids=[20, 21])
    x, y = s.get_batch()
    assert x.shape == (64, 128) and (x[:, 1:] == y[:, :-1]).all()
    a = FimBatchSampler(p, 128, 16, seed=9, spec=SPEC, fim_rate=0.5, line_rate=0.5, newline_ids=[20, 21])
    b = FimBatchSampler(p, 128, 16, seed=9, spec=SPEC, fim_rate=0.5, line_rate=0.5, newline_ids=[20, 21])
    assert all(torch_equal(a.get_batch(), b.get_batch()) for _ in range(3))


def test_line_rate_cero_conserva_el_comportamiento_anterior(stream_file):
    p, _, _ = stream_file
    old = FimBatchSampler(p, 128, 8, seed=5, spec=SPEC, fim_rate=0.5)
    new = FimBatchSampler(p, 128, 8, seed=5, spec=SPEC, fim_rate=0.5, line_rate=0.0, newline_ids=[20, 21])
    assert all(torch_equal(old.get_batch(), new.get_batch()) for _ in range(3))
