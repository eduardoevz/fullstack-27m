"""Amplia el tokenizer con los tokens FIM (Fase 10) sin cambiar ningun id existente.

    python -m src.tokenizer.extend            # tokenizer-16384.json -> tokenizer-v2.json
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from tokenizers import AddedToken, Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tokenizer.pretokenizer import FIM_TOKENS


@dataclass(frozen=True)
class FimIds:
    prefix: int
    suffix: int
    middle: int


def add_fim_tokens(tok: Tokenizer) -> Tokenizer:
    """Devuelve una copia del tokenizer con los tokens FIM al final. Idempotente."""
    new = Tokenizer.from_str(tok.to_str())
    missing = [t for t in FIM_TOKENS if new.token_to_id(t) is None]
    if missing:
        new.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in missing])
    return new


def fim_ids(tok: Tokenizer) -> FimIds:
    ids = [tok.token_to_id(t) for t in FIM_TOKENS]
    if any(i is None for i in ids):
        raise ValueError("el tokenizer no tiene los tokens FIM; usa tokenizer-v2.json (python -m src.tokenizer.extend)")
    return FimIds(*ids)


def main() -> None:
    from src.config import load_config

    cfg = load_config()
    src = Path(cfg.paths.tokenizer_file)
    out = src.with_name("tokenizer-v2.json")
    tok = Tokenizer.from_file(str(src))
    new = add_fim_tokens(tok)
    for t, i in tok.get_vocab().items():
        assert new.token_to_id(t) == i, f"cambio de id en {t!r}"
    new.save(str(out))
    print(f"{src.name}: {tok.get_vocab_size()} -> {out.name}: {new.get_vocab_size()} "
          f"(FIM ids {[new.token_to_id(t) for t in FIM_TOKENS]})")


if __name__ == "__main__":
    main()
