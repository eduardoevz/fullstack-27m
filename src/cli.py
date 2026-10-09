"""Demo interactiva de autocompletado: escribes codigo y el modelo lo continua.

    python src/cli.py                  # o doble clic en demo.bat

Escribe el inicio de tu codigo (varias lineas) y termina con una linea que contenga solo un punto (.).
Modo FIM (modelo de la Fase 10): escribe <CURSOR> donde quieres que el modelo rellene, p. ej.
    def suma(a, b):
    <CURSOR>
    return r
Comandos (en una linea sola):  :lang ts|tsx|js|jsx|py   :temp 0.7   :tokens 200   :help   :quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.sample import frame_fim_prompt, frame_prompt, generate, load_model

LANGS = {"js": "js", "jsx": "js", "ts": "ts", "tsx": "ts", "py": "py"}
REP_PENALTY = 1.1        # Fase 8: duplica la sintaxis valida y casi elimina los bucles (ver avance/fase-08-*.md)
CURSOR_MARK = "<CURSOR>"
HELP = ("Escribe el inicio de tu codigo y termina con una linea que solo tenga un punto (.)\n"
        "  :lang ts|tsx|js|jsx|py   cambia el lenguaje      :temp 0.7    temperatura (0 = determinista)\n"
        "  :tokens 200              largo de la continuacion  :quit        salir")


class Streamer:
    """Imprime el texto a medida que salen tokens, sin partir caracteres UTF-8 a medias."""

    def __init__(self, tok):
        self.tok, self.ids, self.printed = tok, [], 0

    def __call__(self, token_id: int) -> None:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):
            return
        sys.stdout.write(text[self.printed:])
        sys.stdout.flush()
        self.printed = len(text)


def read_block() -> str | None:
    """Lee lineas hasta una linea '.'; devuelve None si hay un comando o EOF."""
    lines = []
    while True:
        try:
            line = input("... " if lines else ">>> ")
        except EOFError:
            return None
        if not lines and line.startswith(":"):
            return line
        if line.strip() == ".":
            return "\n".join(lines) + "\n" if lines else ""
        lines.append(line)


def main() -> None:
    import argparse
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="config del modelo (por defecto v1)")
    ap.add_argument("--ckpt", default=None, help="checkpoint (por defecto best.pt de la carpeta del config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    sys.stdout.reconfigure(encoding="utf-8")
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    eot = tok.token_to_id("<|endoftext|>")
    lang, temp, n_tokens = "tsx", 0.7, 200
    print(f"fullstack-27m (paso {step}) | lenguaje: {lang} | T={temp} | {n_tokens} tokens\n{HELP}\n")

    while True:
        block = read_block()
        if block is None:
            break
        if block.startswith(":"):
            cmd, _, arg = block.partition(" ")
            if cmd in (":quit", ":q"):
                break
            elif cmd == ":lang" and arg.strip() in LANGS:
                lang = arg.strip()
            elif cmd == ":temp" and arg.replace(".", "", 1).isdigit():
                temp = float(arg)
            elif cmd == ":tokens" and arg.strip().isdigit():
                n_tokens = int(arg)
            else:
                print(HELP)
            print(f"lenguaje: {lang} | T={temp} | {n_tokens} tokens")
            continue
        if not block.strip():
            continue
        text = block.rstrip("\n") if not block.endswith("\n\n") else block
        sys.stdout.write("\n--- completando ---\n" + text)
        generate(model, frame_prompt(tok, text, LANGS[lang]), n_tokens, temp, 40, 0.95, eot,
                 on_token=Streamer(tok), repetition_penalty=REP_PENALTY)
        print("\n--- fin ---\n")


if __name__ == "__main__":
    main()
