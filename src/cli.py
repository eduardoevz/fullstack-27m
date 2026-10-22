"""Demo interactiva de autocompletado: escribes codigo y el modelo lo continua.

    python src/cli.py                  # o doble clic en demo.bat

Escribe el inicio de tu codigo (varias lineas) y termina con una linea que contenga solo un punto (.).
Modo FIM (modelos v2): escribe <CURSOR> donde quieres que el modelo rellene, p. ej.
    def suma(a, b):
    <CURSOR>
    return r
Con el modelo final de la Fase 11 (ver demo_v2.bat):
    python src/cli.py --config config/model_v2_long_c.json --ckpt <hito-0018310.pt>
Si hay varios <CURSOR>, se rellena el primero. Sin <CURSOR> el modelo continua el texto (sin emitir marcadores FIM).
Comandos (en una linea sola):  :lang ts|tsx|js|jsx|py   :temp 0.7   :tokens 200   :rep 1.1   :help   :quit
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.sample import fim_banned_ids, frame_fim_prompt, frame_prompt, generate, load_model

LANGS = {"js": "js", "jsx": "js", "ts": "ts", "tsx": "ts", "py": "py"}
REP_PENALTY = 1.1        # Fase 8: duplica la sintaxis valida y casi elimina los bucles (ver avance/fase-08-*.md)
CURSOR_MARK = "<CURSOR>"
HELP = ("Escribe el inicio de tu codigo y termina con una linea que solo tenga un punto (.)\n"
        "Con <CURSOR> en el texto, el modelo rellena ese hueco (modelos v2).\n"
        "  :lang ts|tsx|js|jsx|py   cambia el lenguaje      :temp 0.7    temperatura (0 = determinista)\n"
        "  :tokens 200              largo de la continuacion  :rep 1.1     penalizacion por repeticion\n"
        "  :quit                    salir")


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


def split_cursor(text: str) -> tuple[str, str] | None:
    """(prefijo, sufijo) alrededor del primer <CURSOR>, o None si el texto no lo tiene."""
    i = text.find(CURSOR_MARK)
    return None if i < 0 else (text[:i], text[i + len(CURSOR_MARK):])


def complete(model, tok, text: str, lang: str, n_tokens: int, temp: float, rep_penalty: float, on_token=None) -> str:
    """Rellena el <CURSOR> (FIM) o continua el texto. Devuelve el texto generado.
    En FIM no se vetan los marcadores: el relleno necesita cerrar con <|endoftext|>; en modo normal si, para no ensuciar la salida."""
    eot = tok.token_to_id("<|endoftext|>")
    parts = split_cursor(text)
    if parts is not None:
        prompt, banned = frame_fim_prompt(tok, parts[0], parts[1], LANGS[lang]), None
    else:
        prompt, banned = frame_prompt(tok, text, LANGS[lang]), fim_banned_ids(tok)
    new = generate(model, prompt, n_tokens, temp, 40, 0.95, eot, on_token=on_token,
                   repetition_penalty=rep_penalty, banned_ids=banned)
    return tok.decode(new, skip_special_tokens=False)


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
    ap.add_argument("--rep-penalty", type=float, default=REP_PENALTY, help="penalizacion por repeticion (Fase 11: 1,1 rinde mas a 512 tokens)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    sys.stdout.reconfigure(encoding="utf-8")
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    lang, temp, n_tokens, rep_pen = "tsx", 0.7, 200, args.rep_penalty
    print(f"fullstack-27m (paso {step}) | lenguaje: {lang} | T={temp} | {n_tokens} tokens | rep={rep_pen}\n{HELP}\n")

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
            elif cmd == ":rep" and arg.replace(".", "", 1).isdigit():
                rep_pen = float(arg)
            else:
                print(HELP)
            print(f"lenguaje: {lang} | T={temp} | {n_tokens} tokens | rep={rep_pen}")
            continue
        if not block.strip():
            continue
        text = block.rstrip("\n") if not block.endswith("\n\n") else block
        parts = split_cursor(text)
        sys.stdout.write("\n--- completando ---\n" + (parts[0] if parts else text))
        complete(model, tok, text, lang, n_tokens, temp, rep_pen, on_token=Streamer(tok))
        if parts:
            sys.stdout.write(parts[1])
        print("\n--- fin ---\n")


if __name__ == "__main__":
    main()
