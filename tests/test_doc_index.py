"""Fase 9, paso 4: indice de documentos y reconstruccion de lo que uso v1."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.doc_index import CLEAN_REASONS, index_corpus, summarize_leftover, used_by_v1
from src.data.encode import build_dataset, special_ids
from test_encode import make_raw, tok  # noqa: F401  (fixture)


def test_reconstruye_exactamente_lo_que_uso_build_dataset(tmp_path, tok):
    raw = make_raw(tmp_path)
    meta = build_dataset(raw, tok, tmp_path / "o", train_tokens=60_000, val_tokens=2_000, seed=3, record_repos=True)
    idx = index_corpus(raw, tok, special_ids(tok))
    used = used_by_v1(idx, train_tokens=60_000, val_tokens=2_000, seed=3)
    assert {idx["repo"][i] for i in np.flatnonzero(used)} == set(meta["train_repos"])
    assert int(idx["length"][used].sum()) == meta["train_tokens"]
    assert not used[idx["hold"]].any()


def test_resumen_de_sobrantes(tmp_path, tok):
    raw = make_raw(tmp_path)
    idx = index_corpus(raw, tok, special_ids(tok))
    used = used_by_v1(idx, 60_000, 2_000, 3)
    rep = summarize_leftover(idx, used)
    total = sum(v["tokens"] for v in rep["leftover_by_framework"].values())
    expected = int(idx["length"][~used & ~idx["hold"]].sum())
    assert total == expected
    assert all(0 <= v["clean_tokens"] <= v["tokens"] for v in rep["leftover_by_framework"].values())
    assert CLEAN_REASONS[0] == "ok"
