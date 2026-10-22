"""Nexora: interfaz web local de fullstack-27m. Sin dependencias: servidor de la biblioteca estandar y una sola pagina.

    python -m src.web.server --config config/model_v2_long_c.json --ckpt <hito-0018310.pt>     # o demo_nexora.bat

Rutas: GET / (la pagina) · GET /static/<archivo> · GET /api/info · POST /api/complete (respuesta SSE: token ... done).
Con <CURSOR> en el texto el modelo rellena ese hueco (FIM); sin el, continua el texto sin emitir marcadores FIM.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cli import CURSOR_MARK, LANGS, REP_PENALTY, split_cursor
from src.config import load_config
from src.sample import fim_banned_ids, frame_fim_prompt, frame_prompt, generate, load_model

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
MAX_CHARS = 20000
MAX_BODY = 4 * MAX_CHARS + 1024      # un caracter puede ocupar hasta 4 bytes en UTF-8
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".png": "image/png", ".gif": "image/gif", ".svg": "image/svg+xml"}


def build_prompt(tok, text: str, lang: str) -> tuple[list[int], list[int] | None, bool]:
    """(ids del prompt, ids vetados, es_fim). En FIM no se vetan los marcadores: el relleno necesita cerrar con <|endoftext|>."""
    parts = split_cursor(text)
    if parts is not None:
        return frame_fim_prompt(tok, parts[0], parts[1], LANGS[lang]), None, True
    return frame_prompt(tok, text, LANGS[lang]), fim_banned_ids(tok), False


def _numero(valor, nombre: str, lo: float, hi: float, entero: bool = False):
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise ValueError(f"{nombre} debe ser un numero")
    if entero and valor != int(valor):
        raise ValueError(f"{nombre} debe ser entero")
    if not lo <= valor <= hi:
        raise ValueError(f"{nombre} debe estar entre {lo:g} y {hi:g}")
    return int(valor) if entero else float(valor)


def parse_request(body: dict) -> dict:
    """Valida el cuerpo de /api/complete y completa los valores por defecto. Lanza ValueError con un mensaje legible."""
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text es obligatorio y no puede estar vacio")
    if len(text) > MAX_CHARS:
        raise ValueError(f"text no puede pasar de {MAX_CHARS} caracteres")
    lang = body.get("lang", "tsx")
    if lang not in LANGS:
        raise ValueError(f"lang debe ser uno de: {', '.join(LANGS)}")
    return {"text": text, "lang": lang,
            "temperature": _numero(body.get("temperature", 0.7), "temperature", 0, 1.5),
            "tokens": _numero(body.get("tokens", 200), "tokens", 1, 512, entero=True),
            "rep_penalty": _numero(body.get("rep_penalty", REP_PENALTY), "rep_penalty", 1.0, 1.5)}


class Decoder:
    """Texto nuevo por cada token, sin partir caracteres UTF-8 a medias (como Streamer de cli.py, pero devuelve el texto)."""

    def __init__(self, tok):
        self.tok, self.ids, self.printed = tok, [], 0

    def feed(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tok.decode(self.ids, skip_special_tokens=False)
        if text.endswith("�"):
            return ""
        nuevo, self.printed = text[self.printed:], len(text)
        return nuevo


def make_server(model, tok, info: dict, host: str, port: int) -> ThreadingHTTPServer:
    """Servidor listo para serve_forever(). `info` se devuelve en /api/info. Una sola generacion a la vez."""
    ocupado = threading.Lock()
    info = {"langs": list(LANGS), "cursor": CURSOR_MARK, **info}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _archivo(self, ruta: Path) -> None:
            if not ruta.is_file() or ruta.resolve().parent != STATIC_DIR.resolve():
                return self._json(404, {"error": "no encontrado"})
            data = ruta.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TYPES.get(ruta.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._archivo(STATIC_DIR / "index.html")
            elif path == "/api/info":
                self._json(200, info)
            elif path.startswith("/static/"):
                self._archivo(STATIC_DIR / unquote(path[len("/static/"):]))
            else:
                self._json(404, {"error": "no encontrado"})

        def do_POST(self):
            if self.path != "/api/complete":
                return self._json(404, {"error": "no encontrado"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if not 0 < n <= MAX_BODY:
                    raise ValueError("cuerpo vacio o demasiado grande")
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("el cuerpo debe ser un objeto JSON")
                req = parse_request(body)
            except (ValueError, UnicodeDecodeError) as e:
                return self._json(400, {"error": str(e)})
            if not ocupado.acquire(blocking=False):
                return self._json(429, {"error": "el modelo esta ocupado con otra generacion"})
            try:
                self._generar(req)
            finally:
                ocupado.release()

        def _evento(self, nombre: str, datos: dict) -> None:
            self.wfile.write(f"event: {nombre}\ndata: {json.dumps(datos, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        def _generar(self, req: dict) -> None:
            ids, vetados, es_fim = build_prompt(tok, req["text"], req["lang"])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            dec, t0 = Decoder(tok), time.perf_counter()

            def al_token(token_id: int) -> None:
                texto = dec.feed(token_id)
                if texto:
                    self._evento("token", {"text": texto})      # si el navegador cerro la pestana, esto lanza OSError y corta

            try:
                nuevos = generate(model, ids, req["tokens"], req["temperature"], 40, 0.95, tok.token_to_id("<|endoftext|>"),
                                  on_token=al_token, repetition_penalty=req["rep_penalty"], banned_ids=vetados)
                seg = time.perf_counter() - t0
                self._evento("done", {"tokens": len(nuevos), "seconds": round(seg, 2),
                                      "tok_s": round(len(nuevos) / seg, 1) if seg > 0 else 0.0,
                                      "stopped": len(nuevos) < req["tokens"], "fim": es_fim})
            except OSError:
                pass                                            # cliente desconectado: se aborta la generacion
            except Exception as e:                              # noqa: BLE001 - se informa al navegador en vez de dejarlo colgado
                try:
                    self._evento("error", {"error": f"{type(e).__name__}: {e}"})
                except OSError:
                    pass

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="config del modelo (por defecto v1)")
    ap.add_argument("--ckpt", default=None, help="checkpoint (por defecto best.pt de la carpeta del config)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="no abrir el navegador")
    args = ap.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    card = json.loads((WEB_DIR / "modelcard.json").read_text(encoding="utf-8"))
    info = {"step": step, "params": sum(p.numel() for p in model.parameters()), "card": card}
    srv = make_server(model, tok, info, args.host, args.port)
    url = f"http://{args.host}:{srv.server_address[1]}"
    print(f"Nexora lista en {url}  (modelo del paso {step}). Ctrl+C para salir.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
