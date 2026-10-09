"""Fase 11: vigilancia de FIM y regla de aborto temprano."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.eval.fim_monitor import decide_abort, newline_token_ids, run_monitor


def test_aborto_no_se_evalua_antes_del_paso():
    m = {"loss_mid": 9.0, "stop_rate": 0.0}
    assert decide_abort(m, 2.0, step=500, abort_step=3000, max_ratio=0.85, min_stop=0.5) is None
    assert decide_abort(m, 2.0, step=500, abort_step=0, max_ratio=0.85, min_stop=0.5) is None   # 0 = nunca


def test_aborto_por_perdida_por_parada_o_por_ambas():
    ok = {"loss_mid": 1.6, "stop_rate": 0.7}
    assert decide_abort(ok, 2.0, 3000, 3000, 0.85, 0.5) is None
    bad_loss = decide_abort({"loss_mid": 1.9, "stop_rate": 0.7}, 2.0, 3000, 3000, 0.85, 0.5)
    assert "perdida" in bad_loss and "parada" not in bad_loss
    bad_stop = decide_abort({"loss_mid": 1.6, "stop_rate": 0.1}, 2.0, 3000, 3000, 0.85, 0.5)
    assert "parada" in bad_stop and "perdida" not in bad_stop
    both = decide_abort({"loss_mid": 1.9, "stop_rate": 0.1}, 2.0, 4000, 3000, 0.85, 0.5)
    assert "perdida" in both and "parada" in both


class FakeTok:
    vocab = {"a": 0, "\n": 1, "\n  ": 2, " x": 3}

    def get_vocab(self):
        return self.vocab

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids)


def test_ids_de_salto_de_linea():
    assert newline_token_ids(FakeTok()) == [1, 2]


class ByteTok:
    """Tokenizador de juguete: un id por caracter ASCII; 126=<|file|>, 125=lang, 124/123/122=FIM, 121=eot."""
    ids = {"<|file|>": 126, "<|lang_ts|>": 125, "<|fim_prefix|>": 124, "<|fim_suffix|>": 123,
           "<|fim_middle|>": 122, "<|endoftext|>": 121}

    def token_to_id(self, t):
        return self.ids[t]

    def encode(self, s):
        class E:
            pass
        e = E()
        e.ids = [ord(c) % 100 for c in s]
        return e


def test_run_monitor_mide_perdida_y_parada_y_restaura_el_modo():
    from src.model.gpt import GPT
    cfg = ModelConfig(vocab_size=128, n_layer=2, n_head=2, d_model=32, d_ff=64, block_size=256,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)
    torch.manual_seed(0)
    m = GPT(cfg)
    m.train()
    cases = [{"lang": "ts", "prefix": "ab\ncd\n", "suffix": "ef\n", "middle": "xy\n"}] * 3
    r = run_monitor(m, ByteTok(), cases, fim=True)
    assert r["n"] == 3 and r["loss_mid"] > 0 and 0.0 <= r["stop_rate"] <= 1.0
    assert m.training is True                                   # devuelve el modelo a modo entrenamiento
    r1 = run_monitor(m, ByteTok(), cases, fim=False)
    assert r1["stop_rate"] is None and r1["loss_mid"] > 0
