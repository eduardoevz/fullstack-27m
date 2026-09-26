import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dedup import Deduper, shingles


def make_file(seed: int, n: int = 60) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        a, b = rng.randrange(10**6), rng.randrange(10**6)
        lines.append(f"export const handler{a} = async (req{b}) => {{ return service{a}.run(req{b}, {i}); }};")
    return "\n".join(lines)


def test_shingles_are_5gram():
    assert len(shingles("a b c d e f g")) >= 1
    assert len(shingles("a b c")) == 1  # texto corto: un unico shingle, sin fallar


def test_exact_duplicate_detected():
    d = Deduper(threshold=0.8)
    f = make_file(1)
    assert d.add_if_new(f) is True
    assert d.add_if_new(f) is False


def test_near_duplicate_detected():
    d = Deduper(threshold=0.8)
    f = make_file(2)
    assert d.add_if_new(f) is True
    edited = f.replace("handler", "h", 3)  # unos pocos cambios sobre ~60 lineas
    assert d.add_if_new(edited) is False


def test_distinct_files_kept():
    d = Deduper(threshold=0.8)
    assert d.add_if_new(make_file(3)) is True
    assert d.add_if_new(make_file(4)) is True
    assert len(d) == 2


def test_save_and_load_roundtrip(tmp_path):
    d = Deduper(threshold=0.8)
    f = make_file(5)
    d.add_if_new(f)
    p = tmp_path / "dedup.npz"
    d.save(p)
    d2 = Deduper.load(p)
    assert len(d2) == 1
    assert d2.add_if_new(f) is False


def test_duplicate_detected_after_flush_and_reload(tmp_path):
    d = Deduper(threshold=0.8)
    files = [make_file(s) for s in range(10, 20)]
    assert all(d.add_if_new(f) for f in files)
    d.flush()
    assert d.add_if_new(files[3]) is False            # candidato viene de los arrays ordenados
    p = tmp_path / "d.npz"
    d.save(p)
    d2 = Deduper.load(p)
    assert len(d2) == 10
    assert d2.add_if_new(files[7].replace("handler", "h", 3)) is False
    assert d2.add_if_new(make_file(99)) is True


def test_legacy_pickle_loads(tmp_path):
    import pickle
    d = Deduper(threshold=0.8)
    f = make_file(30)
    d.add_if_new(f)
    p = tmp_path / "old.pkl"
    with open(p, "wb") as fh:
        pickle.dump({"threshold": 0.8, "seed": 1234, "sigs": [d._sigs[0]], "buckets": []}, fh)
    d2 = Deduper.load(p)
    assert len(d2) == 1 and d2.add_if_new(f) is False
