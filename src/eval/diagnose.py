"""Diagnostico de v1: ¿por que el modelo no abre/cierra bien los archivos?

    python src/eval/diagnose.py [--ckpt ...] [--windows 256] [--out benchmarks/v1_diagnosis.json]

Mide sobre val.bin (repos disjuntos de train):
  1. perdida por posicion dentro de ventanas de 512 tokens al azar;
  2. perdida de los primeros 64 tokens en ventanas que empiezan en <|file|> frente a ventanas al azar;
  3. P(<|endoftext|>) que el modelo da al final real de archivos completos (<= 500 tokens);
  4. estadistica del muestreo de entrenamiento (offsets al azar): largo de documentos, fraccion de ventanas con
     un <|endoftext|> y probabilidad de que una ventana empiece en un inicio de documento.

Regla fijada ANTES de medir (classify_hypothesis): la hipotesis «el muestreo por offsets al azar impidio aprender a
abrir/cerrar archivos» se da por confirmada si la mediana de P(eot) < 0,3 y la perdida de los primeros 64 tokens en
inicios de documento es >= 10 % mayor que al azar; descartada si la mediana de P(eot) >= 0,5; mixta en otro caso.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config

BUCKETS = [(0, 16), (16, 64), (64, 256), (256, 512)]
P_EOT_CONFIRM, P_EOT_REJECT, START_RATIO_CONFIRM = 0.3, 0.5, 1.10


def doc_spans(data: np.ndarray, file_id: int, eot_id: int) -> list[tuple[int, int]]:
    """(indice de <|file|>, indice de su <|endoftext|>) por documento enmarcado; ignora un fragmento inicial cortado."""
    file_pos = np.flatnonzero(data == file_id)
    eot_pos = np.flatnonzero(data == eot_id)
    spans = []
    for f in file_pos:
        k = np.searchsorted(eot_pos, f)
        if k < len(eot_pos):
            spans.append((int(f), int(eot_pos[k])))
    return spans


def bucket_means(losses: np.ndarray, buckets: list[tuple[int, int]]) -> dict[str, float]:
    return {f"{a}-{b - 1}": float(np.mean(losses[a:b])) for a, b in buckets}


def window_stats(data: np.ndarray, file_id: int, eot_id: int, window: int) -> dict:
    """Sobre TODOS los offsets posibles de una ventana: fraccion que contiene un eot y fraccion que empieza en un inicio."""
    n_win = len(data) - window
    eot_pos = np.flatnonzero(data == eot_id)
    cover = np.zeros(n_win + 1, dtype=np.int64)
    for e in eot_pos:
        lo, hi = max(0, int(e) - window + 1), min(int(e), n_win - 1)
        if lo <= hi:
            cover[lo] += 1
            cover[hi + 1] -= 1
    with_eot = int((np.cumsum(cover)[:n_win] > 0).sum())
    starts = int((np.flatnonzero(data == file_id) < n_win).sum())
    return {"n_windows": n_win, "share_with_eot": with_eot / n_win, "share_starting_at_doc_start": starts / n_win}


def classify_hypothesis(median_p_eot: float, start_loss_ratio: float) -> str:
    if median_p_eot < P_EOT_CONFIRM and start_loss_ratio >= START_RATIO_CONFIRM:
        return "confirmada"
    if median_p_eot >= P_EOT_REJECT:
        return "descartada"
    return "mixta"


@torch.no_grad()
def _token_losses(model, rows: np.ndarray, batch: int = 16) -> np.ndarray:
    """rows: (N, T+1) -> perdida por token (N, T)."""
    out = []
    for i in range(0, len(rows), batch):
        t = torch.from_numpy(rows[i:i + batch].astype(np.int64))
        logits, _ = model(t[:, :-1])
        out.append(F.cross_entropy(logits.transpose(1, 2), t[:, 1:], reduction="none").numpy())
    return np.concatenate(out)


def main() -> None:
    from tokenizers import Tokenizer

    from src.sample import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--max-doc-tokens", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "v1_diagnosis.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "v1-best.pt", cfg)
    file_id, eot_id = tok.token_to_id("<|file|>"), tok.token_to_id("<|endoftext|>")
    block = cfg.model.block_size
    data = np.memmap(cfg.paths.val_bin, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(args.seed)
    spans = doc_spans(data, file_id, eot_id)

    # 1 y 2: ventanas al azar frente a ventanas que empiezan en un documento
    offs = rng.integers(0, len(data) - block - 1, size=args.windows)
    rand_rows = np.stack([data[o:o + block + 1] for o in offs])
    long_starts = [s for s, _ in spans if s + block + 1 <= len(data)]
    pick = rng.choice(long_starts, size=args.windows, replace=False)
    start_rows = np.stack([data[s:s + block + 1] for s in pick])
    rand_loss, start_loss = _token_losses(model, rand_rows), _token_losses(model, start_rows)
    first64_rand, first64_start = float(rand_loss[:, :64].mean()), float(start_loss[:, :64].mean())
    ratio = first64_start / first64_rand

    # 3: P(eot) en el final real de archivos completos cortos
    short = [(s, e) for s, e in spans if e - s <= args.max_doc_tokens]
    short = [short[i] for i in rng.permutation(len(short))[:300]]
    p_end, p_mid = [], []
    with torch.no_grad():
        for s, e in short:
            ids = torch.from_numpy(data[s:e].astype(np.int64))[None]       # sin el eot final
            probs = torch.softmax(model(ids)[0][0], dim=-1)[:, eot_id]
            p_end.append(float(probs[-1]))
            p_mid.append(float(probs[2:-1].mean()) if len(probs) > 3 else 0.0)
    p_end = np.array(p_end)

    lengths = np.array([e - s + 1 for s, e in spans])
    stats = window_stats(data, file_id, eot_id, block)
    verdict = classify_hypothesis(float(np.median(p_end)), ratio)
    result = {
        "checkpoint_step": step, "windows": args.windows, "val_docs": len(spans),
        "loss_by_position_random_windows": bucket_means(rand_loss.mean(axis=0), BUCKETS),
        "loss_by_position_doc_start_windows": bucket_means(start_loss.mean(axis=0), BUCKETS),
        "first64_loss": {"random_windows": first64_rand, "doc_start_windows": first64_start, "ratio_start_over_random": ratio},
        "p_eot_at_real_end": {"n_docs": len(p_end), "median": float(np.median(p_end)), "mean": float(p_end.mean()),
                              "share_above_0.5": float((p_end > 0.5).mean()), "share_above_0.1": float((p_end > 0.1).mean()),
                              "mean_p_eot_mid_document": float(np.mean(p_mid))},
        "doc_length_tokens": {"median": float(np.median(lengths)), "mean": float(lengths.mean()),
                              "p25": float(np.percentile(lengths, 25)), "p75": float(np.percentile(lengths, 75)),
                              "share_longer_than_window": float((lengths > block).mean()),
                              "share_up_to_256": float((lengths <= 256).mean())},
        "random_window_sampling": stats,
        "rule": {"p_eot_confirm_below": P_EOT_CONFIRM, "p_eot_reject_from": P_EOT_REJECT,
                 "start_loss_ratio_confirm_from": START_RATIO_CONFIRM},
        "hypothesis_no_aprendio_abrir_cerrar_por_el_muestreo": verdict,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
