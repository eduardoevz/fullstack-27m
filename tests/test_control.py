import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.estado import summarize
from src.training.control import consume_stop_file, stop_requested


def test_stop_file_is_detected_and_consumed(tmp_path):
    f = tmp_path / "PARAR.txt"
    assert not stop_requested(f)
    f.write_text("")
    assert stop_requested(f)
    consume_stop_file(f)
    assert not f.exists() and not stop_requested(f)
    consume_stop_file(f)                       # no falla si no existe


def write_log(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["step", "loss", "lr", "tok_s", "rss_gb", "grad_norm", "val_loss"])
        w.writeheader()
        for r in rows:
            w.writerow({"step": "", "loss": "", "lr": "", "tok_s": "", "rss_gb": "", "grad_norm": "",
                        "val_loss": "", **r})


def test_summarize_uses_last_row_per_step_and_estimates_days(tmp_path):
    p = tmp_path / "train.csv"
    write_log(p, [
        {"step": 10, "loss": 9.0, "tok_s": 1000},
        {"step": 20, "loss": 8.0, "tok_s": 1000},
        {"step": 20, "loss": 7.5, "tok_s": 1000},      # paso repetido tras reanudar: manda el ultimo
        {"step": 20, "val_loss": 7.7},
    ])
    s = summarize(p, max_steps=100, tokens_per_step=1000)
    assert s["step"] == 20 and s["loss"] == 7.5 and s["val_loss"] == 7.7
    assert s["remaining_steps"] == 80
    assert abs(s["days_left"] - (80 * 1000 / 1000 / 86400)) < 1e-9     # 80 pasos de 1000 tok a 1000 tok/s


def test_summarize_without_log_returns_none(tmp_path):
    assert summarize(tmp_path / "no.csv", 100, 1000) is None
