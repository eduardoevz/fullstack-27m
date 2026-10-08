"""Fase 9, paso 6: mezcla v2 (pisos por dominio), preferencia por documentos frescos y construccion."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.build_v2 import build_v2, class_key, plan_mix, select_docs_prefer
from src.data.encode import balance_fractions
from src.tokenizer.corpus import is_holdout
from test_encode import TEXTS, tok  # noqa: F401  (fixture)


def share(tokens, fractions, pred):
    kept = {c: min(t, t * fractions[c]) for c, t in tokens.items()}
    tot = sum(kept.values())
    return sum(v for c, v in kept.items() if pred(c)) / tot


S = 1e6
TOKENS = {"react|tsx": 400 * S, "react|other": 300 * S, "django|other": 500 * S, "nextjs|tsx": 40 * S,
          "nextjs|other": 10 * S, "fastapi|other": 60 * S, **{f"x{i}|other": 300 * S for i in range(8)}}
FLOORS = {"nextjs": 0.05, "fastapi": 0.09, "tsx": 0.15}
TARGET = 600 * S


def in_group(name):
    return lambda c: (c.startswith("nextjs") if name == "nextjs" else c.startswith("fastapi") if name == "fastapi" else c.endswith("|tsx"))


def test_class_key():
    assert class_key("nextjs", "tsx") == "nextjs|tsx" and class_key("react", "jsx") == "react|other"


def test_sin_pisos_equivale_a_balance_fractions():
    a = plan_mix(TOKENS, TARGET, floors={})
    b = balance_fractions(TOKENS, TARGET)
    assert a["fractions"].keys() == b.keys()
    assert all(a["fractions"][c] == pytest.approx(b[c], abs=1e-4) for c in b)


def test_pisos_se_cumplen_cuando_hay_material():
    plan = plan_mix(TOKENS, TARGET, FLOORS)
    fr = plan["fractions"]
    assert plan["unmet"] == {} and plan["boost"] > 1.0          # sin refuerzo, fastapi no llegaria al 9 %
    sin = plan_mix(TOKENS, TARGET, {})["fractions"]
    assert share(TOKENS, sin, in_group("fastapi")) < 0.09
    for name, floor in FLOORS.items():
        assert share(TOKENS, fr, in_group(name)) >= floor - 1e-3
    kept = sum(TOKENS[c] * fr[c] for c in TOKENS)
    assert kept == pytest.approx(TARGET, rel=0.01)
    assert all(0 <= f <= 1.0 for f in fr.values())


def test_piso_inalcanzable_se_informa_y_no_rompe():
    plan = plan_mix({"react|other": 1000.0, "nextjs|tsx": 2.0}, 500, {"nextjs": 0.05})
    assert "nextjs" in plan["unmet"] and plan["fractions"]["nextjs|tsx"] == 1.0


def test_todo_cabe_se_conserva_todo():
    plan = plan_mix({"a|other": 10.0, "b|other": 20.0}, 1000, {})
    assert plan["fractions"] == {"a|other": 1.0, "b|other": 1.0}


def test_select_prefiere_frescos_sin_repetir():
    classes = np.zeros(100, dtype=np.int64)
    lengths = np.full(100, 10)
    fresh = np.zeros(100, dtype=bool)
    fresh[:30] = True
    a = select_docs_prefer(classes, lengths, {0: 0.5}, np.random.default_rng(1), fresh)
    assert len(a) == 50 and len(set(a.tolist())) == 50
    assert fresh[a].sum() == 30                      # entran todos los frescos antes que ningun usado
    b = select_docs_prefer(classes, lengths, {0: 0.5}, np.random.default_rng(1), fresh)
    assert np.array_equal(a, b)
    c = select_docs_prefer(classes, lengths, {0: 0.2}, np.random.default_rng(1), fresh)
    assert fresh[c].all() and len(c) == 20


def write_shards(d, rows, name="shard-00000.jsonl.gz"):
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / name, "wt", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def rows_v1(n):
    out = []
    kinds = [("react", "tsx", TEXTS[0], "a.tsx"), ("express", "js", TEXTS[2], "a.js"), ("flask", "py", TEXTS[1], "a.py")]
    for i in range(n):
        fw, lang, code, path = kinds[i % 3]
        out.append({"code": code + f"// v1 {i}\n", "lang": lang, "framework": fw, "path": path, "repo": f"o/v{i}"})
    return out


def rows_modern(n):
    out = []
    for i in range(n):
        out.append({"code": TEXTS[1] + f"# m{i}\n", "lang": "py", "framework": "fastapi", "path": "main.py", "repo": f"o/m{i}"})
    out.append({"code": TEXTS[1] + "# tipos\n", "lang": "ts", "framework": "nestjs", "path": "src/x.d.ts", "repo": "o/dts"})
    header = "// Copyright 2020 X\n// License MIT\n// Permission granted\n// Warranty none\n"
    out.append({"code": header + TEXTS[0], "lang": "tsx", "framework": "nextjs", "path": "app/p.tsx", "repo": "o/lic"})
    return out


def test_build_v2_end_to_end(tmp_path, tok):
    raw, modern = tmp_path / "raw", tmp_path / "raw_modern"
    write_shards(raw, rows_v1(1500))
    write_shards(modern, rows_modern(300))
    fresh = np.zeros(1500, dtype=bool)
    fresh[::2] = True
    meta = build_v2([raw, modern], tok, tmp_path / "o", train_tokens=40_000, val_tokens=2_000, floors={"fastapi": 0.03},
                    fresh_by_source={0: fresh}, domain_val_tokens=1_000, seed=1, record_repos=True)
    train = np.fromfile(tmp_path / "o" / "train.bin", dtype=np.uint16)
    assert meta["train_tokens"] == len(train) and 0 < len(train) <= 40_000
    assert (tmp_path / "o" / "val.bin").exists() and (tmp_path / "o" / "meta.json").exists()
    assert "o/dts" not in meta["train_repos"] + meta["val_repos"]          # limpieza: .d.ts fuera
    assert set(meta["val_repos"]).isdisjoint(meta["train_repos"])
    assert all(is_holdout(r) for r in meta["val_repos"]) and not any(is_holdout(r) for r in meta["train_repos"])
    assert meta["mix"]["share_by_framework"]["fastapi"] >= 0.03
    assert meta["sources"]["modern_docs"] > 0 and meta["sources"]["v1_docs"] > 0
    z = np.load(tmp_path / "o" / "train_docs.npz")
    file_id = tok.token_to_id("<|file|>")
    assert len(z["offset"]) == meta["train_docs"] and int(z["length"].sum()) == len(train)
    assert (train[z["offset"]] == file_id).all()                              # los offsets caen en inicios de documento
    for name in meta["domain_val"]:
        assert (tmp_path / "o" / f"val_{name}.bin").exists()


def test_la_licencia_se_recorta_antes_de_codificar(tmp_path, tok):
    raw = tmp_path / "raw"
    write_shards(raw, [{"code": "// Copyright 2020 X\n// License MIT\n// Permission granted\n// Warranty none\n" + TEXTS[0],
                        "lang": "tsx", "framework": "react", "path": "a.tsx", "repo": "o/x%d" % i} for i in range(5)] * 1)
    build_v2([raw], tok, tmp_path / "o", train_tokens=10_000_000, val_tokens=1_000, floors={}, fresh_by_source={},
             domain_val_tokens=0, seed=0, keep_temp=True)
    all_ids = np.fromfile(tmp_path / "o" / "_all.bin", dtype=np.uint16)
    text = tok.decode(all_ids.tolist(), skip_special_tokens=False)
    assert "Copyright" not in text and "import React" in text


def test_piso_inalcanzable_no_sobrerrefuerza_los_demas_pisos():
    tokens = {"fastapi|other": 14 * S, "react|tsx": 140 * S, "nextjs|tsx": 75 * S, "node|other": 300 * S, "django|other": 400 * S,
              **{f"x{i}|other": 100 * S for i in range(4)}}
    floors = {"fastapi": 0.03, "tsx": 0.15}                    # fastapi: 14/600 = 2,3 % < 3 %, imposible
    plan = plan_mix(tokens, 600 * S, floors)
    fr = plan["fractions"]
    assert "fastapi" in plan["unmet"] and fr["fastapi|other"] == 1.0
    tsx_share = share(tokens, fr, lambda c: c.endswith("|tsx"))
    assert 0.15 - 1e-3 <= tsx_share < 0.28                      # cumple el piso sin meter todo el tsx
    assert fr["react|tsx"] < 1.0
    assert sum(tokens[c] * fr[c] for c in tokens) == pytest.approx(600 * S, rel=0.01)
