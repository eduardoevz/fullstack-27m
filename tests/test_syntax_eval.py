import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.prompts import PROMPTS
from src.eval.syntax import check_fragments, plan_counts, summarize, trim_to_valid

needs_node = pytest.mark.skipif(
    shutil.which("node") is None or not (Path("C:/llm-fullstack-data/tools/node_modules/typescript").exists()),
    reason="falta node o el paquete typescript en C:/llm-fullstack-data/tools",
)


def ok(code, lang):
    return check_fragments([{"id": 0, "lang": lang, "code": code}])[0]["ok"]


# ---------------------------------------------------------------- Python
def test_python_valid_and_invalid():
    assert ok("def f(x):\n    return x + 1\n", "py")
    assert not ok("def f(x:\n    return x\n", "py")
    assert not ok("x = (1,\n", "py")


# ---------------------------------------------------------------- JS / TS (node + typescript)
@needs_node
def test_js_family_valid_and_invalid():
    assert ok("const a = (x) => x * 2;\nexport default a;\n", "js")
    assert ok("const el = <div className='a'>{x}</div>;\n", "jsx")
    assert not ok("function f( {\n", "js")
    assert not ok("const el = <div\n", "jsx")


@needs_node
def test_ts_family_accepts_types_and_rejects_broken_code():
    assert ok("interface A { id: number }\nconst f = (a: A): string => `${a.id}`;\n", "ts")
    assert ok("export default function Nav(): JSX.Element { return <nav>{1}</nav>; }\n", "tsx")
    assert not ok("interface A { id: number\n", "ts")


@needs_node
def test_ts_class_accepts_either_ts_or_tsx_parser():
    # <T>x es un cast valido en .ts pero no en .tsx; un fragmento 'ts' vale si parsea en cualquiera de los dos.
    assert ok("const y = <number>x;\n", "ts")
    assert ok("const el = <div>{x}</div>;\n", "ts")


@needs_node
def test_js_class_accepts_plain_js_and_jsx():
    assert ok("const el = <div>{x}</div>;\n", "js")
    assert ok("var a = 1;\n", "js")


def test_batch_keeps_ids_and_order():
    res = check_fragments([{"id": 7, "lang": "py", "code": "x = 1"}, {"id": 3, "lang": "py", "code": "x ="}])
    assert [r["id"] for r in res] == [7, 3] and [r["ok"] for r in res] == [True, False]


# ---------------------------------------------------------------- recorte al ultimo punto valido
def test_trim_cuts_a_truncated_last_statement():
    code = "import os\n\ndef a():\n    return 1\n\ndef b(x):\n    return os.path.join(x,"
    kept, ratio = trim_to_valid(code, "py")
    assert kept.endswith("return 1\n") or kept.rstrip().endswith("return 1")
    assert 0.5 < ratio < 1.0


def test_trim_of_already_valid_code_keeps_everything():
    code = "a = 1\nb = 2\n"
    kept, ratio = trim_to_valid(code, "py")
    assert kept == code and ratio == 1.0


def test_trim_of_hopeless_code_returns_nothing():
    kept, ratio = trim_to_valid("def (:\n  ??\n", "py")
    assert kept.strip() == "" and ratio == 0.0


# ---------------------------------------------------------------- reparto y resumen
def test_plan_counts_sum_to_n_and_cover_the_three_classes():
    assert plan_counts(100) == {"js": 34, "ts": 33, "py": 33}
    assert sum(plan_counts(7).values()) == 7 and set(plan_counts(100)) == {"js", "ts", "py"}


def test_summarize_computes_rates_per_class_and_overall():
    rows = [
        {"lang": "py", "finished": True, "ok": True, "trimmed_ok": True, "retention": 1.0},
        {"lang": "py", "finished": False, "ok": False, "trimmed_ok": True, "retention": 0.9},
        {"lang": "ts", "finished": False, "ok": False, "trimmed_ok": False, "retention": 0.2},
        {"lang": "ts", "finished": True, "ok": False, "trimmed_ok": False, "retention": 0.0},
    ]
    s = summarize(rows)
    assert s["total"]["n"] == 4 and s["total"]["strict"] == 0.25 and s["total"]["trimmed"] == 0.5
    assert s["total"]["finished_n"] == 2 and s["total"]["finished_strict"] == 0.5
    assert s["py"]["strict"] == 0.5 and s["ts"]["strict"] == 0.0


# ---------------------------------------------------------------- suite de prompts
def test_prompt_suite_has_20_wellformed_unique_prompts():
    assert len(PROMPTS) == 20
    assert len({p["id"] for p in PROMPTS}) == 20
    for p in PROMPTS:
        assert p["lang"] in {"js", "jsx", "ts", "tsx", "py"}
        assert p["prompt"].strip() and p["description"] and p["expect"]


# ---------------------------------------------------------------- repeticion
def test_repeated_line_ratio():
    from src.eval.syntax import repeated_line_ratio
    assert repeated_line_ratio("a = 1\nb = 2\nc = 3\n") == 0.0
    assert repeated_line_ratio("x = foo()\nx = foo()\nx = foo()\ny = 2\n") == 0.5      # 2 de 4 lineas son repeticion
    assert repeated_line_ratio("\n\n  \n") == 0.0                                      # sin lineas con contenido
    assert repeated_line_ratio("}\n}\n}\nx = 1\n") == 0.0                              # llaves sueltas no cuentan


def test_summarize_includes_mean_repetition():
    rows = [{"lang": "py", "finished": True, "ok": True, "trimmed_ok": True, "retention": 1.0, "repeat": 0.2},
            {"lang": "py", "finished": True, "ok": True, "trimmed_ok": True, "retention": 1.0, "repeat": 0.4}]
    s = summarize(rows)
    assert s["total"]["mean_repeat"] == pytest.approx(0.3)
    assert s["total"]["repetitive_share"] == 0.5          # fragmentos con >= 30 % de lineas repetidas
