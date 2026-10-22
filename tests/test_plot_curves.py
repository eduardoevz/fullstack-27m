"""Fase 13: las curvas del v2 unen varios tramos de entrenamiento reanudados, sin filas repetidas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.plot_curves import merge_runs


def row(step, loss="", val=""):
    return {"step": str(step), "loss": str(loss) if loss != "" else "", "val_loss": str(val) if val != "" else ""}


def test_un_solo_tramo_queda_igual():
    rows = [row(10, 1.5), row(20, 1.4), row(20, val=1.45)]
    assert merge_runs([rows]) == rows


def test_el_tramo_posterior_reemplaza_los_pasos_que_solapan():
    a = [row(10, 1.5), row(20, 1.4), row(30, 1.3)]
    b = [row(20, 1.35), row(30, 1.25), row(40, 1.2)]
    out = merge_runs([a, b])
    assert [(r["step"], r["loss"]) for r in out] == [("10", "1.5"), ("20", "1.35"), ("30", "1.25"), ("40", "1.2")]


def test_quita_filas_identicas_repetidas_dentro_de_un_tramo():
    a = [row(10, 1.5), row(10, val=1.6), row(10, 1.5), row(10, val=1.6)]
    assert merge_runs([a]) == [row(10, 1.5), row(10, val=1.6)]
