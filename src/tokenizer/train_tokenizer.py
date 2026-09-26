"""Entrena el BPE byte-level propio. Uso:

    python src/tokenizer/train_tokenizer.py --vocab-size 32768 --sample-mb 700
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import psutil
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tokenizer.pretokenizer import SPECIAL_TOKENS, build_pretokenizer


def train_from_iterator(texts, vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = build_pretokenizer()
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,               # ids 0..5, en este orden
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def main() -> None:
    from src.config import load_config
    from src.tokenizer.corpus import iter_train_texts

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--sample-mb", type=int, default=700)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or Path(cfg.paths.data_root) / f"tokenizer-{args.vocab_size}.json")
    proc = psutil.Process()
    t0 = time.time()
    tok = train_from_iterator(iter_train_texts(cfg.paths.raw_dir, args.sample_mb), args.vocab_size)
    tok.save(str(out))
    print(f"vocab={tok.get_vocab_size()} guardado en {out} | {time.time() - t0:.0f}s | RSS final {proc.memory_info().rss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
