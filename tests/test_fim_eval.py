"""Fase 10: evaluacion de infilling (casos, perdida del medio, puntuacion)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import ModelConfig
from src.eval.fim_eval import build_cases, loss_on_middle, middle_similarity, trim_context

DOC = "".join(f"const v{i} = compute({i});\n" for i in range(40))


def docs(n=20):
    return [{"lang": "ts", "text": DOC.replace("compute", f"compute{k}")} for k in range(n)]


def test_casos_deterministas_y_bien_formados():
    a = build_cases(docs(), n=12, seed=3)
    b = build_cases(docs(), n=12, seed=3)
    assert a == b and len(a) == 12
    assert a != build_cases(docs(), n=12, seed=4)
    for c in a:
        full = c["prefix_full"] + c["middle"] + c["suffix_full"]
        assert full in [d["text"] for d in docs()]
        assert c["prefix_full"].endswith("\n") and c["middle"].endswith("\n")
        assert 1 <= c["middle"].count("\n") <= 3
        assert c["prefix_full"].count("\n") >= 3 and c["suffix_full"].count("\n") >= 2


def test_documentos_cortos_se_omiten():
    short = [{"lang": "py", "text": "a = 1\nb = 2\n"}]
    assert build_cases(short, n=5, seed=0) == []


def test_trim_context_respeta_el_presupuesto_y_conserva_lo_cercano():
    pre = "".join(f"linea {i}\n" for i in range(100))
    suf = "".join(f"fin {i}\n" for i in range(100))
    count = lambda s: len(s.split())            # tokens de juguete: palabras
    p, s = trim_context(pre, suf, count, max_prefix=40, max_suffix=20)
    assert count(p) <= 40 and count(s) <= 20
    assert pre.endswith(p) and suf.startswith(s)               # recorta por el lado lejano
    assert p.endswith("linea 99\n") and s.startswith("fin 0\n")


def test_similitud_del_medio():
    assert middle_similarity("x = 1\n", "x = 1\n") == {"exact": 1.0, "similarity": 1.0}
    r = middle_similarity("  x = 1  \n", "x = 1\n")
    assert r["exact"] == 1.0                                    # ignora espacios en los extremos
    r = middle_similarity("y = 2\n", "x = 1\n")
    assert r["exact"] == 0.0 and 0.0 <= r["similarity"] < 1.0


def test_perdida_del_medio_solo_cuenta_los_tokens_del_medio():
    from src.model.gpt import GPT
    cfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, d_model=32, d_ff=64, block_size=64,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
    torch.manual_seed(0)
    m = GPT(cfg).eval()
    prompt, mid = list(range(5, 25)), [30, 31, 32]
    loss, n = loss_on_middle(m, prompt, mid)
    assert n == 3 and 0 < loss < 20
    with torch.no_grad():
        logits, _ = m(torch.tensor([prompt + mid]))
    lp = torch.log_softmax(logits[0], -1)
    expect = -sum(lp[len(prompt) - 1 + i, t] for i, t in enumerate(mid)) / 3
    assert loss == pytest.approx(float(expect), abs=1e-5)
    assert loss_on_middle(m, [5] * 80, mid) is None            # no cabe en el contexto
