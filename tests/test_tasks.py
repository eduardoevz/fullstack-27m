import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.tasks import (TASKS, run_task, truncate_completion, unsafe_js, unsafe_python)

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="falta node")


# ---------------------------------------------------------------- recorte de la continuacion
def test_truncate_python_stops_at_the_next_top_level_statement():
    assert truncate_completion("    return x\n\ndef other():\n    pass\n", "py") == "    return x\n"
    assert truncate_completion("    a = 1\n    return a\n", "py") == "    a = 1\n    return a\n"


def test_truncate_python_keeps_blank_lines_inside_the_body():
    assert truncate_completion("    a = 1\n\n    return a\nprint(1)", "py") == "    a = 1\n\n    return a\n"


def test_truncate_js_stops_after_the_closing_brace_at_column_zero():
    assert truncate_completion("  return a + b;\n}\n\nconsole.log(1)", "js") == "  return a + b;\n}"
    assert truncate_completion("  if (x) {\n    return 1;\n  }\n  return 2;\n}\nmore", "js") == "  if (x) {\n    return 1;\n  }\n  return 2;\n}"
    assert truncate_completion("  return 1;\n", "js") == "  return 1;\n"          # nunca cerro: se deja tal cual


# ---------------------------------------------------------------- filtros de seguridad (antes de ejecutar nada)
@pytest.mark.parametrize("code", [
    "import os\nos.remove('x')",
    "import subprocess",
    "from shutil import rmtree",
    "x = open('f', 'w')",
    "eval('1+1')",
    "exec('a=1')",
    "__import__('os')",
    "x = ().__class__.__bases__",
    "getattr(obj, 'a')",
    "import socket",
])
def test_unsafe_python_is_rejected(code):
    assert unsafe_python(code) is not None


@pytest.mark.parametrize("code", [
    "import math\nx = math.floor(1.5)",
    "import re\nre.sub('a', 'b', s)",
    "from collections import Counter\nc = Counter(s)",
    "def f(x):\n    return sorted(set(x))\n",
])
def test_safe_python_is_accepted(code):
    assert unsafe_python(code) is None


@pytest.mark.parametrize("code", [
    "const fs = require('fs');",
    "process.exit(1)",
    "fetch('http://x')",
    "eval('1')",
    "new Function('return 1')()",
    "import('fs')",
    "globalThis.process",
    "const c = require('child_process')",
])
def test_unsafe_js_is_rejected(code):
    assert unsafe_js(code) is not None


def test_safe_js_is_accepted():
    assert unsafe_js("  return arr.filter((x, i) => arr.indexOf(x) === i);\n}") is None


# ---------------------------------------------------------------- las tareas en si
def test_there_are_30_tasks_with_20_python_and_10_js_and_unique_ids():
    assert len(TASKS) == 30
    assert sum(t["lang"] == "py" for t in TASKS) == 20 and sum(t["lang"] == "js" for t in TASKS) == 10
    assert len({t["id"] for t in TASKS}) == 30
    for t in TASKS:
        assert t["prompt"].strip() and t["tests"].strip() and t["reference"].strip()


@needs_node
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_reference_solution_passes_its_own_tests(task):
    ok, reason = run_task(task, task["reference"])
    assert ok, reason


@needs_node
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_an_empty_or_wrong_solution_fails(task):
    wrong = "    return None\n" if task["lang"] == "py" else "  return undefined;\n}"
    ok, _ = run_task(task, wrong)
    assert not ok


def test_python_infinite_loop_is_killed_by_the_timeout():
    task = next(t for t in TASKS if t["lang"] == "py")
    ok, reason = run_task(task, "    while True:\n        pass\n", timeout=1.0)
    assert not ok and "timeout" in reason


def test_unsafe_completion_is_not_executed(tmp_path):
    task = next(t for t in TASKS if t["lang"] == "py")
    marker = tmp_path / "pwned.txt"
    ok, reason = run_task(task, f"    import os\n    open(r'{marker}', 'w').write('x')\n")
    assert not ok and "unsafe" in reason and not marker.exists()


@needs_node
def test_node_permission_model_blocks_writes_even_if_the_static_screen_is_bypassed(tmp_path):
    from src.eval.tasks import _run_js
    marker = tmp_path / "pwned.txt"
    program = f"const fs = process.getBuiltinModule('fs'); fs.writeFileSync({str(marker)!r}, 'x');"
    ok, _ = _run_js(program, timeout=5.0)
    assert not ok and not marker.exists()


# ---------------------------------------------------------------- resumen pass@1
def test_summarize_tasks_computes_pass_at_1_with_ci_per_language():
    from src.eval.run_tasks import summarize_tasks
    rows = [{"id": "py-a", "lang": "py", "passed": True}, {"id": "py-a", "lang": "py", "passed": False},
            {"id": "py-b", "lang": "py", "passed": False}, {"id": "py-b", "lang": "py", "passed": False},
            {"id": "js-c", "lang": "js", "passed": True}, {"id": "js-c", "lang": "js", "passed": True}]
    s = summarize_tasks(rows, seed=0)
    assert s["n_tasks"] == 3 and s["n_samples"] == 6
    assert s["pass_at_1"]["mean"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert s["py"]["pass_at_1"]["mean"] == pytest.approx(0.25) and s["js"]["pass_at_1"]["mean"] == 1.0
    assert s["pass_at_1"]["lo"] <= s["pass_at_1"]["mean"] <= s["pass_at_1"]["hi"]
    assert s["tasks_solved_at_least_once"] == 2
