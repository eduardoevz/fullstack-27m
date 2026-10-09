"""Fase 10: campos opcionales del config (FIM, init_from) y config de la prueba v2."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, load_config


def test_el_config_v1_no_cambia_y_los_campos_nuevos_tienen_defecto():
    c = load_config()
    assert c.training.fim_rate == 0.0 and c.training.init_from == ""
    assert c.model.vocab_size == 16384 and c.model.param_count() == 20_453_760


def test_config_v2():
    c = load_config(REPO_ROOT / "config" / "model_v2.json")
    assert c.model.vocab_size == 16387
    assert c.model.param_count() == 20_453_760 + 3 * 384
    t = c.training
    assert t.fim_rate == 0.5 and t.init_from.endswith("v1-best.pt")
    assert (t.max_steps, t.warmup_steps, t.learning_rate, t.min_learning_rate) == (1000, 100, 3e-4, 3e-5)
    assert c.paths.token_dir.endswith("tokens_v2") and c.paths.tokenizer_file.endswith("tokenizer-v2.json")
    assert c.paths.checkpoint_dir.endswith("checkpoints_v2")
    base = load_config()                         # mismo tamano de batch y contexto que v1
    assert (c.model.block_size, t.micro_batch_size, t.grad_accum_steps) == \
        (base.model.block_size, base.training.micro_batch_size, base.training.grad_accum_steps)


def test_clave_desconocida_sigue_rechazada(tmp_path):
    raw = json.loads((REPO_ROOT / "config" / "model_27m.json").read_text(encoding="utf-8"))
    raw["training"]["no_existe"] = 1
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)


def test_falta_una_clave_obligatoria_sigue_fallando(tmp_path):
    raw = json.loads((REPO_ROOT / "config" / "model_27m.json").read_text(encoding="utf-8"))
    del raw["training"]["max_steps"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
