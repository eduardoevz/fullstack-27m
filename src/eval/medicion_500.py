"""Espera a que el entrenamiento llegue a un paso multiplo de 500, conserva su checkpoint,
lo evalua con 200 casos (fim_eval + fim_boundary) y escribe un resumen de salud.

Uso:
    python -m src.eval.medicion_500 --step 12500
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "model_v2_long_c.json"
LOG = REPO_ROOT / "logs" / "train_v2_long_c.csv"
CKPT_DIR = Path("C:/llm-fullstack-data/checkpoints_v2_long_c")
HITOS = Path("C:/llm-fullstack-data/checkpoints_v2_long_c_hitos")
STOP_LIMIT, RATIO_LIMIT, V1_REF = 0.25, 0.88, 2.5955


def leer_csv() -> list[dict]:
    with open(LOG, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def vigilancia(step: int) -> dict | None:
    """Fila de val y fila de FIM del paso `step`, o None si aun no estan las dos."""
    val = mid = None
    for r in leer_csv():
        if r["step"] != str(step):
            continue
        if r["val_loss"]:
            val = r
        if r["fim_mid_loss"]:
            mid = r
    return {"val": val, "mid": mid} if val and mid else None


def correr(modulo: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", modulo, *args], cwd=REPO_ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    step = ap.parse_args().step

    while (v := vigilancia(step)) is None:
        time.sleep(30)

    HITOS.mkdir(exist_ok=True)
    hito = HITOS / f"hito-{step:07d}.pt"
    if not hito.exists():
        shutil.copy2(CKPT_DIR / f"ckpt-{step:07d}.pt", hito)

    ev, fr = REPO_ROOT / "benchmarks" / f"fim_eval_{step}.json", REPO_ROOT / "benchmarks" / f"fim_boundary_{step}.json"
    correr("src.eval.fim_eval", "--config-fim", str(CONFIG), "--ckpt-fim", str(hito), "--out", str(ev))
    correr("src.eval.fim_boundary", "--config", str(CONFIG), "--ckpt", str(hito), "--out", str(fr))
    s = json.loads(ev.read_text(encoding="utf-8"))["summary"]
    b = json.loads(fr.read_text(encoding="utf-8"))

    ventana = [r for r in leer_csv() if r["loss"] and step - 500 < int(r["step"]) <= step]
    loss = [float(r["loss"]) for r in ventana]
    gn = [float(r["grad_norm"]) for r in ventana]
    tok = [float(r["tok_s"]) for r in ventana]
    mid, stop = float(v["mid"]["fim_mid_loss"]), float(v["mid"]["fim_stop_rate"])

    print(json.dumps({
        "paso": step,
        "val_loss": float(v["val"]["val_loss"]), "val_fim": float(v["val"]["val_fim_loss"]),
        "fim_medio": mid, "fim_ratio_monitor": round(mid / V1_REF, 3), "limite_ratio": RATIO_LIMIT,
        "tasa_parada": stop, "limite_parada": STOP_LIMIT,
        "eval200": {"ratio": round(s["loss_ratio"], 3), "similitud": round(s["similarity_fim"], 3),
                    "parsea": s["parses_fim"], "exacto": s["exact_fim"]},
        "frontera_texto": round(b["texto"]["ratio"], 3), "frontera_token": round(b["token"]["ratio"], 3),
        "loss_train_media": round(sum(loss) / len(loss), 4), "gnorm_max": max(gn),
        "tok_s_medio": round(sum(tok) / len(tok)),
    }, indent=1))


if __name__ == "__main__":
    main()
