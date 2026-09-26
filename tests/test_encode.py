import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.encode import (balance_fractions, build_dataset, frame_doc, select_docs,
                             special_ids, write_split)
from src.tokenizer.corpus import is_holdout
from src.tokenizer.train_tokenizer import train_from_iterator

TEXTS = [
    "import React from 'react';\nexport function App() {\n    return <div>hola</div>;\n}\n",
    "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/x')\nasync def x():\n    return {'a': 1}\n",
    "const app = express();\napp.get('/x', (req, res) => {\n\tres.json({ok: true});\n});\n",
]


@pytest.fixture(scope="module")
def tok():
    return train_from_iterator(iter(TEXTS * 80), vocab_size=400)


def test_frame_doc_prefix_and_suffix(tok):
    sp = special_ids(tok)
    ids = np.array([50, 51, 52], dtype=np.uint16)
    out = frame_doc(ids, "tsx", sp)
    assert out.dtype == np.uint16
    assert out.tolist() == [tok.token_to_id("<|file|>"), tok.token_to_id("<|lang_ts|>"), 50, 51, 52, tok.token_to_id("<|endoftext|>")]
    assert frame_doc(ids, "jsx", sp)[1] == tok.token_to_id("<|lang_js|>")


def test_balance_fractions_caps_big_classes_and_keeps_small():
    fr = balance_fractions({"a": 100, "b": 10, "c": 10}, target=60)
    assert fr["b"] == 1.0 and fr["c"] == 1.0
    assert abs(fr["a"] * 100 + 10 + 10 - 60) <= 1.0          # K = 40


def test_balance_fractions_keeps_everything_if_under_target():
    assert balance_fractions({"a": 5, "b": 7}, target=100) == {"a": 1.0, "b": 1.0}


def test_select_docs_respects_budget_is_deterministic_and_unique():
    classes = np.array([0] * 1000 + [1] * 100)
    lengths = np.full(1100, 10)
    fr = {0: 0.3, 1: 1.0}
    a = select_docs(classes, lengths, fr, np.random.default_rng(7))
    b = select_docs(classes, lengths, fr, np.random.default_rng(7))
    assert np.array_equal(a, b) and len(set(a.tolist())) == len(a)
    assert lengths[a][classes[a] == 0].sum() <= 0.3 * 10000 + 1e-9
    assert (classes[a] == 1).sum() == 100                     # la clase pequena entra entera
    assert not np.array_equal(a, select_docs(classes, lengths, fr, np.random.default_rng(8)))


def test_write_split_concatenates_in_order(tmp_path):
    flat = np.arange(100, dtype=np.uint16)
    p = tmp_path / "all.bin"
    flat.tofile(p)
    mm = np.memmap(p, dtype=np.uint16, mode="r")
    offsets, lengths = np.array([0, 10, 50]), np.array([5, 20, 10])
    out = tmp_path / "o.bin"
    n = write_split(mm, offsets, lengths, np.array([2, 0, 1]), out, block_docs=2)
    assert n == 35
    assert np.fromfile(out, dtype=np.uint16).tolist() == list(range(50, 60)) + list(range(0, 5)) + list(range(10, 30))


def make_raw(tmp_path, n_docs=3000):
    raw = tmp_path / "raw"
    raw.mkdir()
    kinds = [("react", "tsx", TEXTS[0], "a.tsx"), ("fastapi", "py", TEXTS[1], "a.py"), ("express", "js", TEXTS[2], "a.js")]
    with gzip.open(raw / "shard-00000.jsonl.gz", "wt", encoding="utf-8", newline="") as f:
        for i in range(n_docs):
            fw, lang, code, path = kinds[0] if i % 5 else kinds[i % 3]      # react domina
            f.write(json.dumps({"code": code + f"// {i}\n", "lang": lang, "framework": fw, "path": path, "repo": f"o/r{i}"}) + "\n")
    return raw


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_build_dataset_end_to_end(tmp_path, tok):
    raw = make_raw(tmp_path)
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    meta = build_dataset(raw, tok, out1, train_tokens=60_000, val_tokens=2_000, seed=3, record_repos=True)
    build_dataset(raw, tok, out2, train_tokens=60_000, val_tokens=2_000, seed=3)
    assert sha(out1 / "train.bin") == sha(out2 / "train.bin") and sha(out1 / "val.bin") == sha(out2 / "val.bin")   # determinismo
    assert not (out1 / "_all.bin").exists()

    train = np.fromfile(out1 / "train.bin", dtype=np.uint16)
    val = np.fromfile(out1 / "val.bin", dtype=np.uint16)
    assert meta["train_tokens"] == len(train) and meta["val_tokens"] == len(val)
    assert 0 < len(val) <= 2_000
    assert len(train) <= 60_000 and len(train) > 40_000
    assert int(train.max()) < tok.get_vocab_size() and int(val.max()) < tok.get_vocab_size()
    eot, file_id = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|file|>")
    assert int((train == eot).sum()) == meta["train_docs"]
    assert int((train == file_id).sum()) == meta["train_docs"]
    assert train[0] == file_id and train[-1] == eot
    # el reequilibrio recorta a la clase dominante sin duplicar
    assert meta["classes"]["react"]["train_after_tokens"] < meta["classes"]["react"]["train_before_tokens"]
    assert meta["classes"]["fastapi"]["train_after_tokens"] == meta["classes"]["fastapi"]["train_before_tokens"]
    # val y train no comparten repos
    assert set(meta["val_repos"]).isdisjoint(meta["train_repos"])
    assert all(is_holdout(r) for r in meta["val_repos"]) and not any(is_holdout(r) for r in meta["train_repos"])
