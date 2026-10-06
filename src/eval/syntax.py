"""Validez sintactica del codigo generado.

Python: `ast.parse`. JS/JSX/TS/TSX: el parser de TypeScript 5.x (via node, `check_syntax.js`).
Mide SOLO sintaxis: una propiedad duplicada en una interface o una variable sin declarar
no son errores sintacticos y pasan. Sirve para "el modelo escribe codigo bien formado",
no para "el codigo funciona".

El modelo tiene tres etiquetas de lenguaje (js, ts, py); `jsx` y `tsx` comparten la de js y ts.
Un fragmento de clase 'js' se acepta si parsea como JS o como JSX; uno de clase 'ts', como TS o TSX.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

CHECK_JS = Path(__file__).with_name("check_syntax.js")
TOOLS_DIR = Path("C:/llm-fullstack-data/tools")
RETENTION_MIN = 0.8          # un fragmento recortado cuenta como valido si conserva >= 80 % de sus lineas
_PARSERS = {"js": ("js", "jsx"), "jsx": ("jsx",), "ts": ("ts", "tsx"), "tsx": ("tsx",)}


def _node_check(items: list[dict]) -> list[dict]:
    env = {**os.environ, "NODE_PATH": str(TOOLS_DIR / "node_modules")}
    proc = subprocess.run(["node", str(CHECK_JS)], input=json.dumps(items).encode("utf-8"),
                          capture_output=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"node fallo: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return json.loads(proc.stdout.decode("utf-8"))


def _py_check(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except (SyntaxError, ValueError, RecursionError, MemoryError) as e:
        return False, f"{type(e).__name__}: {e}"


def check_fragments(items: list[dict]) -> list[dict]:
    """items: [{"id", "lang", "code"}]. Devuelve [{"id", "ok", "error"}] en el mismo orden."""
    results: dict[int, dict] = {}
    node_jobs, expand = [], {}
    for pos, it in enumerate(items):
        if it["lang"] == "py":
            ok, err = _py_check(it["code"])
            results[pos] = {"id": it["id"], "ok": ok, "error": err}
        else:
            expand[pos] = []
            for kind in _PARSERS[it["lang"]]:
                expand[pos].append(len(node_jobs))
                node_jobs.append({"id": len(node_jobs), "lang": kind, "code": it["code"]})
    if node_jobs:
        verdicts = _node_check(node_jobs)
        for pos, idxs in expand.items():
            vs = [verdicts[i] for i in idxs]
            good = next((v for v in vs if v["ok"]), None)
            results[pos] = {"id": items[pos]["id"], "ok": good is not None,
                            "error": "" if good else vs[0]["error"]}
    return [results[i] for i in range(len(items))]


def _nonempty(lines: list[str]) -> int:
    return sum(1 for l in lines if l.strip())


def trim_to_valid(code: str, lang: str) -> tuple[str, float]:
    """Prefijo mas largo (en lineas) que parsea, y la fraccion de lineas no vacias que conserva.
    Un fragmento generado se corta a la fuerza al llegar al limite de tokens; esto separa
    «mal escrito» de «cortado a mitad de una sentencia»."""
    lines = code.splitlines(keepends=True)
    total = _nonempty(lines)
    prefixes = ["".join(lines[:n]) for n in range(len(lines), -1, -1)]
    verdicts = check_fragments([{"id": i, "lang": lang, "code": p} for i, p in enumerate(prefixes)])
    for p, v in zip(prefixes, verdicts):
        if v["ok"]:
            return p, (_nonempty(p.splitlines()) / total if total else 0.0)
    return "", 0.0


REPETITIVE_MIN = 0.3         # un fragmento es «repetitivo» si >= 30 % de sus lineas son copias de otras


def repeated_line_ratio(code: str) -> float:
    """Fraccion de lineas con contenido (mas de 3 caracteres, para ignorar llaves sueltas) que repiten otra."""
    lines = [l.strip() for l in code.splitlines() if len(l.strip()) > 3]
    return 1 - len(set(lines)) / len(lines) if lines else 0.0


def plan_counts(n: int) -> dict[str, int]:
    base, extra = divmod(n, 3)
    return {lang: base + (1 if i < extra else 0) for i, lang in enumerate(("js", "ts", "py"))}


def _rate(rows: list[dict], key: str) -> float | None:
    return sum(1 for r in rows if r[key]) / len(rows) if rows else None


def _block(rows: list[dict]) -> dict:
    finished = [r for r in rows if r["finished"]]
    block = {
        "n": len(rows),
        "strict": _rate(rows, "ok"),
        "trimmed": _rate(rows, "trimmed_ok"),
        "finished_n": len(finished),
        "finished_strict": _rate(finished, "ok"),
        "mean_retention": sum(r["retention"] for r in rows) / len(rows) if rows else None,
    }
    reps = [r["repeat"] for r in rows if "repeat" in r]
    block["mean_repeat"] = sum(reps) / len(reps) if reps else None
    block["repetitive_share"] = sum(1 for x in reps if x >= REPETITIVE_MIN) / len(reps) if reps else None
    return block


def summarize(rows: list[dict]) -> dict:
    out = {"total": _block(rows)}
    for lang in sorted({r["lang"] for r in rows}):
        out[lang] = _block([r for r in rows if r["lang"] == lang])
    return out
