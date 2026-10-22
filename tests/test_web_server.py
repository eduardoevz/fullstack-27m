"""Fase 14 (Nexora): el servidor web local arma el prompt (FIM o normal), valida la entrada y transmite la generacion por SSE."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import CURSOR_MARK
from src.sample import fim_banned_ids
from src.tokenizer.extend import add_fim_tokens
from src.web.server import Decoder, build_prompt, make_server, parse_request
from test_cli import tiny_for
from test_encode import tok  # noqa: F401  (fixture)


# ---------------------------------------------------------------- build_prompt
def test_build_prompt_con_cursor_es_fim_y_no_veta_marcadores(tok):
    t = add_fim_tokens(tok)
    ids, banned, es_fim = build_prompt(t, f"const a = {CURSOR_MARK};\n", "ts")
    assert es_fim and banned is None
    assert t.token_to_id("<|fim_middle|>") == ids[-1]


def test_build_prompt_sin_cursor_veta_los_marcadores_fim(tok):
    t = add_fim_tokens(tok)
    ids, banned, es_fim = build_prompt(t, "const a = 1;\n", "ts")
    assert not es_fim
    assert banned == fim_banned_ids(t) and banned
    assert t.token_to_id("<|fim_middle|>") not in ids


# ---------------------------------------------------------------- parse_request
def test_parse_request_aplica_valores_por_defecto():
    r = parse_request({"text": "x = 1"})
    assert r == {"text": "x = 1", "lang": "tsx", "temperature": 0.7, "tokens": 200, "rep_penalty": 1.1}


@pytest.mark.parametrize("body", [
    {}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": "a" * 20001},
    {"text": "a", "lang": "rust"},
    {"text": "a", "temperature": -0.1}, {"text": "a", "temperature": 1.6}, {"text": "a", "temperature": "x"},
    {"text": "a", "tokens": 0}, {"text": "a", "tokens": 513}, {"text": "a", "tokens": 1.5},
    {"text": "a", "rep_penalty": 0.9}, {"text": "a", "rep_penalty": 1.6},
    {"text": "a", "temperature": True},
])
def test_parse_request_rechaza_entradas_invalidas(body):
    with pytest.raises(ValueError):
        parse_request(body)


def test_parse_request_acepta_los_limites():
    r = parse_request({"text": "a", "lang": "py", "temperature": 0, "tokens": 512, "rep_penalty": 1.5})
    assert r["lang"] == "py" and r["temperature"] == 0.0 and r["tokens"] == 512


# ---------------------------------------------------------------- Decoder
def test_decoder_no_emite_caracteres_utf8_partidos(tok):
    ids = tok.encode("café ñandú 日本").ids
    d = Decoder(tok)
    out = "".join(d.feed(i) for i in ids)
    assert "�" not in out
    assert out == tok.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------- servidor real
@pytest.fixture
def servidor(tok):
    t = add_fim_tokens(tok)
    srv = make_server(tiny_for(t), t, {"step": 7, "params": 123}, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def post(url, body):
    req = urllib.request.Request(url + "/api/complete", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30)


def eventos(resp):
    """Lista de (evento, datos) de un flujo SSE."""
    out, ev = [], None
    for raw in resp.read().decode("utf-8").splitlines():
        if raw.startswith("event: "):
            ev = raw[7:]
        elif raw.startswith("data: "):
            out.append((ev, json.loads(raw[6:])))
    return out


def test_complete_transmite_tokens_y_termina_con_done(servidor):
    resp = post(servidor, {"text": "const a = 1;\n", "lang": "ts", "temperature": 0, "tokens": 10})
    assert resp.headers["Content-Type"].startswith("text/event-stream")
    evs = eventos(resp)
    assert evs[-1][0] == "done"
    assert evs[-1][1]["tokens"] <= 10 and "tok_s" in evs[-1][1] and "stopped" in evs[-1][1]
    assert all(e == "token" for e, _ in evs[:-1])
    assert all("text" in d for _, d in evs[:-1])


def test_complete_en_modo_fim_funciona(servidor):
    evs = eventos(post(servidor, {"text": f"const a = {CURSOR_MARK};\n", "lang": "ts", "tokens": 8}))
    assert evs[-1][0] == "done" and evs[-1][1]["fim"] is True


def test_complete_con_entrada_invalida_responde_400(servidor):
    with pytest.raises(urllib.error.HTTPError) as e:
        post(servidor, {"text": ""})
    assert e.value.code == 400
    assert "error" in json.loads(e.value.read())


def test_complete_con_json_roto_responde_400(servidor):
    req = urllib.request.Request(servidor + "/api/complete", data=b"{no es json")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_api_info_devuelve_la_informacion_del_modelo(servidor):
    info = json.loads(urllib.request.urlopen(servidor + "/api/info", timeout=10).read())
    assert info["step"] == 7 and info["params"] == 123 and "tsx" in info["langs"]


def test_index_y_estaticos_se_sirven_y_rutas_desconocidas_dan_404(servidor):
    html = urllib.request.urlopen(servidor + "/", timeout=10).read().decode("utf-8")
    assert "Nexora" in html
    for ruta in ("/static/app.css", "/static/app.js"):
        assert urllib.request.urlopen(servidor + ruta, timeout=10).status == 200
    for ruta in ("/nada", "/static/../server.py", "/static/%2e%2e/server.py"):
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(servidor + ruta, timeout=10)
        assert e.value.code == 404
