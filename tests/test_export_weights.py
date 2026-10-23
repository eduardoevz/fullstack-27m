"""Publicacion: el export deja solo los pesos (sin optimizador ni RNG), cargables con weights_only=True y por load_model."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.export_weights import export_weights
from src.sample import load_model
from test_generation import TINY, tiny


def fake_training_ckpt(path):
    m = tiny()
    torch.save({"step": 18310, "best_val": 1.36, "model": m.state_dict(),
                "optimizer": {"state": {0: torch.zeros(1000)}}, "sampler": {"i": 3}, "rng": {"x": 1}}, path)
    return m


def test_export_deja_solo_pesos_y_metadatos(tmp_path):
    fake_training_ckpt(tmp_path / "ckpt.pt")
    info = export_weights(tmp_path / "ckpt.pt", tmp_path / "out.pt")
    out = torch.load(tmp_path / "out.pt", map_location="cpu", weights_only=True)
    assert set(out) == {"model", "step"} and out["step"] == 18310
    assert info["step"] == 18310 and info["tensors"] == len(out["model"])


def test_export_es_mas_pequeno_y_el_modelo_cargado_da_los_mismos_logits(tmp_path):
    m = fake_training_ckpt(tmp_path / "ckpt.pt")
    export_weights(tmp_path / "ckpt.pt", tmp_path / "out.pt")
    assert (tmp_path / "out.pt").stat().st_size < (tmp_path / "ckpt.pt").stat().st_size
    loaded, step = load_model(tmp_path / "out.pt", SimpleNamespace(model=TINY))
    x = torch.randint(0, TINY.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.equal(m(x)[0], loaded(x)[0])
    assert step == 18310
