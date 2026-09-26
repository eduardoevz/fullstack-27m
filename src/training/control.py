"""Parada limpia sin ventana: si existe el archivo PARAR.txt, el entrenamiento guarda y sale."""

from __future__ import annotations

from pathlib import Path


def stop_requested(path: str | Path) -> bool:
    return Path(path).exists()


def consume_stop_file(path: str | Path) -> None:
    Path(path).unlink(missing_ok=True)
