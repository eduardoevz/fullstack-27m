"""Entrenamiento del modelo en CPU, reanudable de forma exacta.

    python src/train.py                 # empieza (o continua si ya hay checkpoints con --resume)
    python src/train.py --resume        # continua desde el ultimo checkpoint
    Ctrl+C  o  parar.bat                # termina el paso en curso, guarda checkpoint y sale
"""

from __future__ import annotations

import argparse
import csv
import signal
import statistics
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import torch

from src.config import REPO_ROOT, load_config
from src.model.gpt import GPT
from src.training.control import consume_stop_file, stop_requested
from src.training.checkpoint import latest_checkpoint, load_checkpoint, rotate_checkpoints, save_checkpoint
from src.training.data import BatchSampler
from src.training.loop import evaluate, run_step
from src.training.optim import build_optimizer

CSV_FIELDS = ["step", "loss", "lr", "tok_s", "rss_gb", "grad_norm", "val_loss"]
STOP_FILE = REPO_ROOT / "PARAR.txt"   # crear este archivo (parar.bat) = guardar y salir
VAL_SEED = 1234        # mismo conjunto de batches de validacion en cada evaluacion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", action="store_true", help="continuar desde el ultimo checkpoint")
    ap.add_argument("--stop-at", type=int, default=None, help="parar tras este paso (pruebas); el LR sigue el config")
    ap.add_argument("--ckpt-dir", default=None, help="carpeta de checkpoints (por defecto la del config)")
    ap.add_argument("--log", default=str(REPO_ROOT / "logs" / "train.csv"))
    ap.add_argument("--compile", action="store_true", help="probar torch.compile")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, mc = cfg.training, cfg.model
    torch.set_num_threads(tc.num_threads)
    torch.manual_seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir or cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(mc)
    optimizer = build_optimizer(model, tc)
    sampler = BatchSampler(cfg.paths.train_bin, mc.block_size, tc.micro_batch_size, seed=args.seed)
    step, best_val = 0, float("inf")
    if args.resume:
        last = latest_checkpoint(ckpt_dir)
        if last is None:
            sys.exit(f"--resume: no hay checkpoints en {ckpt_dir}")
        meta = load_checkpoint(last, model, optimizer, sampler)
        step, best_val = meta["step"], meta["best_val"]
        print(f"Reanudado desde {last.name}: paso {step}, mejor val {best_val:.4f}")
    elif latest_checkpoint(ckpt_dir):
        sys.exit(f"Ya hay checkpoints en {ckpt_dir}. Usa --resume, o otra --ckpt-dir, para no pisarlos.")

    fwd = torch.compile(model) if args.compile else model
    end = min(tc.max_steps, args.stop_at) if args.stop_at else tc.max_steps

    consume_stop_file(STOP_FILE)                   # un PARAR.txt viejo no debe frenar esta sesion
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    def checkpoint(is_best: bool = False) -> None:
        save_checkpoint(ckpt_dir / f"ckpt-{step:07d}.pt", model, optimizer, sampler, step, best_val)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, sampler, step, best_val)
        rotate_checkpoints(ckpt_dir, tc.keep_last_checkpoints)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    proc = psutil.Process()
    tokens_per_step = tc.tokens_per_step(mc.block_size)
    recent, baseline = deque(maxlen=20), 0.0
    acc_loss = acc_gnorm = 0.0
    t_win = time.perf_counter()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, CSV_FIELDS)
        if new_log:
            w.writeheader()
        n_win = 0
        while step < end and not stop["now"] and not stop_requested(STOP_FILE):
            t0 = time.perf_counter()
            loss, gnorm = run_step(fwd, optimizer, sampler, tc, step)
            step += 1
            recent.append(tokens_per_step / (time.perf_counter() - t0))
            acc_loss, acc_gnorm, n_win = acc_loss + loss, acc_gnorm + gnorm, n_win + 1

            if step % tc.log_interval == 0:
                tok_s = n_win * tokens_per_step / (time.perf_counter() - t_win)
                row = {"step": step, "loss": f"{acc_loss / n_win:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.3e}",
                       "tok_s": f"{tok_s:.0f}", "rss_gb": f"{proc.memory_info().rss / 2**30:.2f}",
                       "grad_norm": f"{acc_gnorm / n_win:.3f}", "val_loss": ""}
                # Guardia termica: mediana movil de tok/s frente a su mejor valor.
                med = statistics.median(recent)
                baseline = max(baseline, med) if len(recent) == recent.maxlen else baseline
                if baseline and len(recent) == recent.maxlen and med < 0.7 * baseline:
                    print(f"[AVISO termico] tok/s {med:.0f} < 70% de {baseline:.0f}: posible throttling")
                w.writerow(row); f.flush()
                print(f"paso {step}/{tc.max_steps}  loss {row['loss']}  lr {row['lr']}  {row['tok_s']} tok/s  "
                      f"RAM {row['rss_gb']} GB  gnorm {row['grad_norm']}", flush=True)
                acc_loss = acc_gnorm = 0.0
                n_win, t_win = 0, time.perf_counter()

            is_best = False
            if step % tc.eval_interval == 0:
                val = evaluate(model, BatchSampler(cfg.paths.val_bin, mc.block_size, tc.micro_batch_size, VAL_SEED),
                               tc.eval_batches)
                is_best = val < best_val
                best_val = min(best_val, val)
                w.writerow({"step": step, "val_loss": f"{val:.4f}"}); f.flush()
                print(f"  >> val loss {val:.4f} (mejor {best_val:.4f})", flush=True)
                t_win = time.perf_counter()
            if step % tc.checkpoint_interval == 0 or is_best:
                checkpoint(is_best)
                t_win = time.perf_counter()

    checkpoint()                                   # al terminar, con Ctrl+C o con PARAR.txt
    stopped = stop["now"] or stop_requested(STOP_FILE)
    consume_stop_file(STOP_FILE)
    print(f"{'Detenido' if stopped else 'Terminado'} en el paso {step}. Checkpoint guardado.")


if __name__ == "__main__":
    main()
