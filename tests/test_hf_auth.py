import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.hf_auth import auth_headers, load_hf_token, read_env_file


def test_read_env_file_ignora_comentarios_y_comillas(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# comentario\n\nA=1\nB = "dos"\nC=\'tres\'\nsinigual\n', encoding="utf-8")
    assert read_env_file(f) == {"A": "1", "B": "dos", "C": "tres"}


def test_read_env_file_inexistente(tmp_path):
    assert read_env_file(tmp_path / "no-existe") == {}


def test_token_prefiere_entorno_sobre_archivo(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=del_archivo\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "del_entorno")
    assert load_hf_token(f) == "del_entorno"


def test_token_desde_archivo_y_ausente(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    f = tmp_path / ".env"
    f.write_text("HF_TOKEN=abc\n", encoding="utf-8")
    assert load_hf_token(f) == "abc"
    assert load_hf_token(tmp_path / "otro") is None


def test_auth_headers(monkeypatch):
    assert auth_headers("abc") == {"Authorization": "Bearer abc"}
    assert auth_headers(None) == {}
