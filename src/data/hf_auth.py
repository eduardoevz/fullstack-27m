"""Token de Hugging Face: se lee del entorno o de un .env local (nunca versionado)."""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("\"'")
    return out


def load_hf_token(env_file: Path = ENV_FILE) -> str | None:
    return os.environ.get("HF_TOKEN") or read_env_file(env_file).get("HF_TOKEN")


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}
