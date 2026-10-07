"""30 tareas funcionales pequenas (20 Python, 10 JS) al estilo HumanEval, con tests unitarios.

El prompt es la firma + docstring (o comentario); el modelo escribe el cuerpo; se ejecuta `prompt + cuerpo + tests`.
Mide si el codigo FUNCIONA en casos concretos; no es un benchmark general: son 30 tareas escritas por nosotros.

SEGURIDAD: esto ejecuta codigo generado por un modelo. Mitigaciones (no un sandbox perfecto):
  - filtro estatico antes de ejecutar: Python con `ast` (imports solo de una lista segura, sin open/eval/exec/getattr ni
    atributos dunder); JS sin require/import/process/fetch/eval/Function/globalThis. Lo rechazado cuenta como fallo y NO se ejecuta;
  - subproceso aislado (`python -I`; `node --permission` sin escritura ni procesos hijos) en una carpeta temporal y con tiempo limite;
  - los tests y las soluciones de referencia son nuestros; solo la continuacion es del modelo.
La red no esta bloqueada por el modelo de permisos de node; por eso el filtro estatico de JS es estricto.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile

SAFE_IMPORTS = {"math", "re", "itertools", "functools", "collections", "string", "json", "typing", "operator",
                "heapq", "bisect", "datetime", "statistics", "textwrap", "urllib", "copy", "decimal", "fractions"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "__import__", "input", "getattr", "setattr", "delattr", "globals",
                "locals", "vars", "breakpoint", "exit", "quit", "memoryview", "help"}
BANNED_JS = ["require(", "import(", "import ", "process.", "process[", "fetch(", "eval(", "Function(", "globalThis",
             "child_process", "XMLHttpRequest", "WebSocket", "constructor", "__proto__"]


def unsafe_python(code: str) -> str | None:
    """Motivo si el codigo Python no debe ejecutarse; None si pasa el filtro."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return "syntax"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in SAFE_IMPORTS:
                    return f"import {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in SAFE_IMPORTS or node.level:
                return f"from {node.module} import"
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            return f"name {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"attribute {node.attr}"
    return None


def unsafe_js(code: str) -> str | None:
    for token in BANNED_JS:
        if token in code:
            return token
    return None


def truncate_completion(completion: str, lang: str) -> str:
    """Corta la continuacion donde acaba la funcion: Python, en la primera linea sin sangria; JS, en la primera `}` en columna 0."""
    if lang == "py":
        m = re.search(r"\n(?=\S)", completion)
        return completion if not m else completion[:m.start() + 1].rstrip("\n") + "\n"
    m = re.search(r"^\}", completion, re.M)
    return completion if not m else completion[:m.end()]


def _env() -> dict:
    return {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}


def _run_py(program: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        open(path, "w", encoding="utf-8").write(program)
        try:
            p = subprocess.run([sys.executable, "-I", "-B", path], cwd=tmp, capture_output=True, timeout=timeout, env=_env())
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return p.returncode == 0, "" if p.returncode == 0 else p.stderr.decode("utf-8", "replace")[-300:]


def _run_js(program: str, timeout: float) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.js")
        open(path, "w", encoding="utf-8").write(program)
        try:
            p = subprocess.run(["node", "--permission", f"--allow-fs-read={tmp}", path], cwd=tmp, capture_output=True,
                               timeout=timeout, env=_env())
        except subprocess.TimeoutExpired:
            return False, "timeout"
    return p.returncode == 0, "" if p.returncode == 0 else p.stderr.decode("utf-8", "replace")[-300:]


def run_task(task: dict, completion: str, timeout: float = 5.0) -> tuple[bool, str]:
    """Ejecuta prompt + continuacion + tests. Devuelve (paso, motivo)."""
    body = truncate_completion(completion, task["lang"])
    if task["lang"] == "py":
        reason = unsafe_python(task["prompt"] + body)
        if reason == "syntax":
            return False, "syntax"
        if reason:
            return False, f"unsafe: {reason}"
        return _run_py(task["prompt"] + body + "\n\n" + task["tests"], timeout)
    reason = unsafe_js(body)
    if reason:
        return False, f"unsafe: {reason}"
    return _run_js(task["prompt"] + body + "\n" + task["tests"], timeout)


def _py(id, doc_sig, tests, reference):
    return {"id": id, "lang": "py", "prompt": doc_sig, "tests": tests, "reference": reference}


def _js(id, sig, tests, reference):
    return {"id": id, "lang": "js", "prompt": sig, "tests": "const assert = require('node:assert');\n" + tests,
            "reference": reference}


TASKS = [
    _py("py-sum-list", 'def sum_list(nums):\n    """Return the sum of all numbers in the list (0 for an empty list)."""\n',
        "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0\nassert sum_list([-1, 1]) == 0",
        "    total = 0\n    for n in nums:\n        total += n\n    return total\n"),
    _py("py-is-even", 'def is_even(n):\n    """Return True if n is even, otherwise False."""\n',
        "assert is_even(4) is True\nassert is_even(7) is False\nassert is_even(0) is True",
        "    return n % 2 == 0\n"),
    _py("py-reverse-string", 'def reverse_string(s):\n    """Return the string s reversed."""\n',
        "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
        "    return s[::-1]\n"),
    _py("py-count-vowels", 'def count_vowels(s):\n    """Count the vowels (a, e, i, o, u, case-insensitive) in s."""\n',
        "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0",
        "    return sum(1 for c in s.lower() if c in 'aeiou')\n"),
    _py("py-factorial", 'def factorial(n):\n    """Return n! for a non-negative integer n."""\n',
        "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800",
        "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n"),
    _py("py-fibonacci", 'def fibonacci(n):\n    """Return the n-th Fibonacci number (fibonacci(0) == 0, fibonacci(1) == 1)."""\n',
        "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
        "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    _py("py-is-palindrome", 'def is_palindrome(s):\n    """Return True if s reads the same backwards, ignoring case and non-alphanumeric characters."""\n',
        "assert is_palindrome('A man, a plan, a canal: Panama') is True\nassert is_palindrome('hello') is False\nassert is_palindrome('') is True",
        "    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n"),
    _py("py-max-in-list", 'def max_in_list(nums):\n    """Return the largest number in the list, or None if the list is empty."""\n',
        "assert max_in_list([3, 9, 2]) == 9\nassert max_in_list([]) is None\nassert max_in_list([-5, -2]) == -2",
        "    if not nums:\n        return None\n    best = nums[0]\n    for n in nums:\n        if n > best:\n            best = n\n    return best\n"),
    _py("py-unique", 'def unique(items):\n    """Return a list with the duplicates removed, keeping the first occurrence order."""\n',
        "assert unique([1, 2, 1, 3, 2]) == [1, 2, 3]\nassert unique([]) == []\nassert unique(['a', 'a']) == ['a']",
        "    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n"),
    _py("py-chunk-list", 'def chunk_list(lst, size):\n    """Split lst into consecutive chunks of the given size (the last one may be shorter)."""\n',
        "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk_list([], 3) == []\nassert chunk_list([1, 2], 5) == [[1, 2]]",
        "    return [lst[i:i + size] for i in range(0, len(lst), size)]\n"),
    _py("py-flatten", 'def flatten(nested):\n    """Flatten a list of lists by one level."""\n',
        "assert flatten([[1, 2], [3], []]) == [1, 2, 3]\nassert flatten([]) == []\nassert flatten([[1]]) == [1]",
        "    return [x for sub in nested for x in sub]\n"),
    _py("py-word-count", 'def word_count(text):\n    """Return a dict mapping each lowercase word in text (split on whitespace) to how many times it appears."""\n',
        "assert word_count('a b a') == {'a': 2, 'b': 1}\nassert word_count('') == {}\nassert word_count('Hi hi') == {'hi': 2}",
        "    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n"),
    _py("py-capitalize-words", 'def capitalize_words(s):\n    """Capitalize the first letter of every space-separated word in s."""\n',
        "assert capitalize_words('hello world') == 'Hello World'\nassert capitalize_words('') == ''\nassert capitalize_words('a') == 'A'",
        "    return ' '.join(w.capitalize() for w in s.split(' '))\n"),
    _py("py-is-valid-email", 'def is_valid_email(s):\n    """Return True if s has exactly one @, a non-empty local part and a domain with a dot that is not at its start or end."""\n',
        "assert is_valid_email('a@b.com') is True\nassert is_valid_email('ab.com') is False\nassert is_valid_email('a@b') is False\n"
        "assert is_valid_email('@b.com') is False\nassert is_valid_email('a@@b.com') is False\nassert is_valid_email('a@.com') is False\nassert is_valid_email('a@b.') is False",
        "    if s.count('@') != 1:\n        return False\n    local, domain = s.split('@')\n"
        "    return bool(local) and '.' in domain and not domain.startswith('.') and not domain.endswith('.')\n"),
    _py("py-clamp", 'def clamp(x, lo, hi):\n    """Limit x to the range [lo, hi]."""\n',
        "assert clamp(5, 0, 10) == 5\nassert clamp(-3, 0, 10) == 0\nassert clamp(99, 0, 10) == 10",
        "    return max(lo, min(x, hi))\n"),
    _py("py-parse-query-string", 'def parse_query_string(qs):\n    """Parse a query string like "a=1&b=2" (an optional leading "?" is ignored) into a dict of strings."""\n',
        "assert parse_query_string('a=1&b=2') == {'a': '1', 'b': '2'}\nassert parse_query_string('') == {}\nassert parse_query_string('?x=y') == {'x': 'y'}",
        "    qs = qs.lstrip('?')\n    out = {}\n    for part in qs.split('&'):\n        if part:\n            k, _, v = part.partition('=')\n            out[k] = v\n    return out\n"),
    _py("py-slugify", 'def slugify(text):\n    """Lowercase text and replace every run of non-alphanumeric characters with a single "-" (no leading or trailing "-")."""\n',
        "assert slugify('Hello, World!') == 'hello-world'\nassert slugify('  a  b ') == 'a-b'\nassert slugify('') == ''",
        "    import re\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n"),
    _py("py-merge-dicts", 'def merge_dicts(a, b):\n    """Return a new dict with the items of a and b; values from b win. Do not modify a or b."""\n',
        "a = {'x': 1}\nb = {'x': 2, 'y': 3}\nassert merge_dicts(a, b) == {'x': 2, 'y': 3}\nassert a == {'x': 1} and b == {'x': 2, 'y': 3}\nassert merge_dicts({}, {}) == {}",
        "    out = dict(a)\n    out.update(b)\n    return out\n"),
    _py("py-paginate", 'def paginate(items, page, per_page):\n    """Return the items of the given 1-indexed page (an empty list if the page is out of range)."""\n',
        "assert paginate(list(range(10)), 2, 3) == [3, 4, 5]\nassert paginate(list(range(10)), 4, 3) == [9]\nassert paginate(list(range(10)), 5, 3) == []",
        "    start = (page - 1) * per_page\n    return items[start:start + per_page]\n"),
    _py("py-gcd", 'def gcd(a, b):\n    """Return the greatest common divisor of two non-negative integers."""\n',
        "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(0, 9) == 9",
        "    while b:\n        a, b = b, a % b\n    return a\n"),
    _js("js-sum", "// Return the sum of all numbers in the array (0 for an empty array).\nfunction sum(arr) {\n",
        "assert.strictEqual(sum([1, 2, 3]), 6);\nassert.strictEqual(sum([]), 0);",
        "  let total = 0;\n  for (const n of arr) total += n;\n  return total;\n}"),
    _js("js-capitalize", "// Return the string with its first letter in uppercase (empty string stays empty).\nfunction capitalize(str) {\n",
        "assert.strictEqual(capitalize('hello'), 'Hello');\nassert.strictEqual(capitalize(''), '');",
        "  if (!str) return '';\n  return str[0].toUpperCase() + str.slice(1);\n}"),
    _js("js-is-even", "// Return true if n is even.\nfunction isEven(n) {\n",
        "assert.strictEqual(isEven(4), true);\nassert.strictEqual(isEven(7), false);",
        "  return n % 2 === 0;\n}"),
    _js("js-uniq", "// Return a new array without duplicates, keeping the first occurrence order.\nfunction uniq(arr) {\n",
        "assert.deepStrictEqual(uniq([1, 2, 1, 3, 2]), [1, 2, 3]);\nassert.deepStrictEqual(uniq([]), []);",
        "  return [...new Set(arr)];\n}"),
    _js("js-chunk", "// Split the array into consecutive chunks of the given size (the last one may be shorter).\nfunction chunk(arr, size) {\n",
        "assert.deepStrictEqual(chunk([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]);\nassert.deepStrictEqual(chunk([], 3), []);",
        "  const out = [];\n  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));\n  return out;\n}"),
    _js("js-range", "// Return the integers from start (inclusive) to end (exclusive).\nfunction range(start, end) {\n",
        "assert.deepStrictEqual(range(2, 5), [2, 3, 4]);\nassert.deepStrictEqual(range(3, 3), []);",
        "  const out = [];\n  for (let i = start; i < end; i++) out.push(i);\n  return out;\n}"),
    _js("js-parse-query", "// Parse a query string like 'a=1&b=2' (an optional leading '?' is ignored) into an object of strings.\nfunction parseQuery(qs) {\n",
        "assert.deepStrictEqual(parseQuery('a=1&b=2'), { a: '1', b: '2' });\nassert.deepStrictEqual(parseQuery(''), {});\nassert.deepStrictEqual(parseQuery('?x=y'), { x: 'y' });",
        "  const out = {};\n  for (const part of qs.replace(/^\\?/, '').split('&')) {\n    if (!part) continue;\n    const [k, v = ''] = part.split('=');\n    out[k] = v;\n  }\n  return out;\n}"),
    _js("js-clamp", "// Limit x to the range [lo, hi].\nfunction clamp(x, lo, hi) {\n",
        "assert.strictEqual(clamp(5, 0, 10), 5);\nassert.strictEqual(clamp(-3, 0, 10), 0);\nassert.strictEqual(clamp(99, 0, 10), 10);",
        "  return Math.max(lo, Math.min(x, hi));\n}"),
    _js("js-group-by", "// Group the objects of arr by the value of the given key. Return an object of arrays.\nfunction groupBy(arr, key) {\n",
        "assert.deepStrictEqual(groupBy([{ t: 'a', n: 1 }, { t: 'b', n: 2 }, { t: 'a', n: 3 }], 't'), "
        "{ a: [{ t: 'a', n: 1 }, { t: 'a', n: 3 }], b: [{ t: 'b', n: 2 }] });\nassert.deepStrictEqual(groupBy([], 't'), {});",
        "  const out = {};\n  for (const item of arr) {\n    (out[item[key]] = out[item[key]] || []).push(item);\n  }\n  return out;\n}"),
    _js("js-flatten", "// Flatten an array of arrays by one level.\nfunction flatten(arr) {\n",
        "assert.deepStrictEqual(flatten([1, [2, 3], [4]]), [1, 2, 3, 4]);\nassert.deepStrictEqual(flatten([]), []);",
        "  return arr.flat();\n}"),
]
