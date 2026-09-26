import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.tokenizer.pretokenizer import build_pretokenizer, SPECIAL_TOKENS, lang_token
from src.tokenizer.train_tokenizer import train_from_iterator

SNIPPETS = [
    "import React, { useState } from 'react';\n\nexport function App() {\n    const [count, setCount] = useState(0);\n"
    "    if (count === 3 && count !== 4) {\n        return count?.value ?? 0;\n    }\n    return <div onClick={() => setCount(count + 1)} />;\n}\n",
    "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/users/{user_id}')\nasync def get_user_by_id(user_id: int):\n"
    "    if user_id == 0:\n        raise ValueError('bad')\n    return {'id': user_id}\n",
    "const app = express();\napp.get('/x', async (req, res) => {\n\tconst getUserById = await db.find(req.params.id);\n\tres.json(getUserById);\n});\n",
]
CORPUS = SNIPPETS * 60


@pytest.fixture(scope="module")
def tok():
    return train_from_iterator(iter(CORPUS), vocab_size=600)


def pieces(text):
    pt = build_pretokenizer()
    return [p for p, _ in pt.pre_tokenize_str(text)]


def test_compound_operators_are_units():
    p = pieces("a === b !== c => d ?. e ?? f :: g")
    for op in ("===", "!==", "=>", "?.", "??", "::"):
        assert any(x.lstrip("Ġ") == op for x in p), (op, p)


def test_identifiers_not_split():
    p = pieces("x = getUserById(snake_case_name)")
    assert "ĠgetUserById" in p
    assert "snake_case_name" in p


def test_indentation_is_one_pretoken():
    p = pieces("if x:\n        y = 1")
    assert "Ċ" + "Ġ" * 8 in p


def test_special_tokens_have_fixed_ids(tok):
    for i, name in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(name) == i
    assert lang_token("tsx") == "<|lang_ts|>" and lang_token("jsx") == "<|lang_js|>" and lang_token("py") == "<|lang_py|>"


@pytest.mark.parametrize("text", [
    SNIPPETS[0], SNIPPETS[1], SNIPPETS[2],
    "\tif (x) {\r\n\t\treturn 1;\r\n\t}\r\n",
    "const s = 'héllo wörld 日本語 🚀';\n",
    "    \n\n   trailing spaces   \n",
    "x=1;y=2;z=x+y//comment\n",
    "",
])
def test_roundtrip_exact(tok, text):
    assert tok.decode(tok.encode(text).ids) == text


def test_indentation_learned_as_single_token(tok):
    assert len(tok.encode("\n        ").ids) <= 2
    assert len(tok.encode("\n    ").ids) <= 2


def test_special_token_text_is_atomic(tok):
    ids = tok.encode("<|file|>x").ids
    assert ids[0] == tok.token_to_id("<|file|>")


def test_holdout_is_deterministic_and_about_one_percent():
    from src.tokenizer.corpus import is_holdout
    repos = [f"user{i}/repo{i}" for i in range(20000)]
    flags = [is_holdout(r) for r in repos]
    assert flags == [is_holdout(r) for r in repos]
    assert 0.005 < sum(flags) / len(flags) < 0.02


def test_train_sample_excludes_holdout_and_respects_size(tmp_path):
    import gzip, json
    from src.tokenizer.corpus import is_holdout, iter_train_texts
    docs = [{"code": f"x{i} = {i}\n" * 20, "lang": "py", "framework": "flask", "path": "a.py", "repo": f"o/r{i}"} for i in range(4000)]
    with gzip.open(tmp_path / "shard-00000.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    total = sum(len(d["code"].encode()) for d in docs)
    (tmp_path / "manifest.json").write_text(json.dumps({"totals": {"kept_bytes": total}}))
    got = list(iter_train_texts(tmp_path, sample_mb=total / 2e6))
    held = {d["code"] for d in docs if is_holdout(d["repo"])}
    assert not (set(got) & held)
    assert 0.35 * len(docs) < len(got) < 0.65 * len(docs)


def test_config_matches_chosen_tokenizer():
    """El vocab_size del modelo debe coincidir con el tokenizer elegido en la Fase 2."""
    from tokenizers import Tokenizer
    cfg = load_config()
    path = Path(cfg.paths.tokenizer_file)
    if not path.exists():
        pytest.skip("tokenizer no entrenado en esta maquina")
    assert Tokenizer.from_file(str(path)).get_vocab_size() == cfg.model.vocab_size
