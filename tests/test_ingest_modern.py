"""Fase 9: seleccion de documentos de la fuente moderna (funcion pura, sin red)."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.dedup import Deduper
from src.data.ingest_modern import WANTED, select_document

CFG = load_config().data


def _next(i=0):
    return ("import Link from 'next/link';\nimport Image from 'next/image';\n"
            + "\n".join(f"export default function Page{i}_{j}() {{ return <Link href='/p{i}{j}'>ir {j}</Link>; }}" for j in range(8)))


def _fast(i=0):
    return ("from fastapi import FastAPI\napp = FastAPI()\n\n"
            + "\n".join(f"@app.get('/r{i}_{j}')\nasync def r{i}_{j}():\n    return {{'id': {j}}}\n" for j in range(8)))


def run(code, path, caps=None, used=None, dedup=None):
    stats, used = Counter(), used if used is not None else Counter()
    rec = select_document(code, path, "u/repo", CFG, dedup if dedup is not None else Deduper(0.8), caps or {}, used, stats)
    return rec, stats, used


def test_acepta_nextjs_y_fastapi():
    rec, _, _ = run(_next(), "app/page.tsx")
    assert rec["framework"] == "nextjs" and rec["lang"] == "tsx" and rec["repo"] == "u/repo"
    rec, _, _ = run(_fast(), "main.py")
    assert rec["framework"] == "fastapi" and rec["lang"] == "py"


def test_descarta_frameworks_que_no_se_buscan():
    flask = "from flask import Flask\napp = Flask(__name__)\n" + "\n".join(f"@app.route('/{i}')\ndef f{i}():\n    return '{i}'\n" for i in range(8))
    rec, stats, _ = run(flask, "app.py")
    assert rec is None and stats["rej_unwanted"] == 1
    assert "nextjs" in WANTED and "fastapi" in WANTED and "flask" not in WANTED


def test_limpieza_se_aplica():
    rec, stats, _ = run(_next(), "types/page.d.ts")
    assert rec is None and stats["rej_dts"] == 1
    header = "// Copyright 2020 X\n// License MIT\n// Permission granted\n// Warranty none\n"
    rec, _, _ = run(header + _next(), "app/page.tsx")
    assert rec["code"] == _next()


def test_duplicado_contra_el_indice():
    dedup = Deduper(0.8)
    assert run(_next(), "a.tsx", dedup=dedup)[0] is not None
    rec, stats, _ = run(_next(), "b.tsx", dedup=dedup)
    assert rec is None and stats["rej_duplicate"] == 1


def test_tope_de_bytes_por_framework():
    caps = {"nextjs": 100}
    rec, stats, used = run(_next(1), "a.tsx", caps=caps, used=Counter({"nextjs": 100}))
    assert rec is None and stats["rej_cap"] == 1
    rec, _, used = run(_next(2), "b.tsx", caps=caps)
    assert rec is not None and used["nextjs"] == len(rec["code"].encode())


def test_react_solo_cuenta_si_es_tsx():
    react_ts = "import React from 'react';\n" + "\n".join(f"export const C{i} = () => <div>{i}</div>;" for i in range(12))
    assert run(react_ts, "a.tsx")[0]["framework"] == "react"
    rec, stats, _ = run(react_ts.replace("<div>", "").replace("</div>", ""), "a.js")
    assert rec is None and stats["rej_unwanted"] == 1
