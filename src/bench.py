"""Banco de pruebas del proyecto: establece la linea base de rendimiento.

Todas las decisiones de dimensionamiento del proyecto se justifican con numeros
salidos de aqui, no con estimaciones. Se ejecuta en la Fase 0 para fijar la linea
base del hardware y se vuelve a ejecutar tras la Fase 4, cuando ya existe el
modelo, para medir su throughput real.

    python src/bench.py                      # linea base completa
    python src/bench.py --quick              # version corta
    python src/bench.py --no-model           # solo hardware
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import REPO_ROOT, load_config  # noqa: E402

# Derate empirico por throttling termico en laptop: el rendimiento sostenido a lo
# largo de dias es inferior al de una rafaga de segundos. Se aplica a las
# proyecciones de duracion para que el presupuesto no sea optimista.
THERMAL_DERATE = 0.7


def _time_it(fn, warmup: int, iters: int) -> float:
    """Segundos por iteracion, descartando el calentamiento."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def bench_gemm(size: int, dtype: torch.dtype, iters: int) -> float:
    """GFLOPS de una multiplicacion densa cuadrada."""
    a = torch.randn(size, size, dtype=dtype)
    b = torch.randn(size, size, dtype=dtype)
    seconds = _time_it(lambda: a @ b, warmup=3, iters=iters)
    return 2 * size ** 3 / seconds / 1e9


def hardware_section(quick: bool) -> dict:
    iters = 5 if quick else 12
    size = 1024
    max_threads = psutil.cpu_count(logical=True) or 8
    physical = psutil.cpu_count(logical=False) or 4
    original = torch.get_num_threads()

    print("\n=== HARDWARE ===")
    print(f"CPU        : {platform.processor()}")
    print(f"Nucleos    : {physical} fisicos / {max_threads} logicos")
    print(f"RAM        : {psutil.virtual_memory().total / 1e9:.1f} GB "
          f"({psutil.virtual_memory().available / 1e9:.1f} GB disponibles)")
    print(f"Vectorizado: {torch.backends.cpu.get_cpu_capability()}   "
          f"MKLDNN: {torch.backends.mkldnn.is_available()}")

    torch.set_num_threads(physical)
    print(f"\n--- GEMM {size}x{size} con {physical} hilos ---")
    gemm = {}
    for label, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        gflops = bench_gemm(size, dtype, iters)
        gemm[label] = round(gflops, 1)
        print(f"  {label:9s} {gflops:7.1f} GFLOPS")

    # Decision sobre precision mixta: en CPUs sin AVX512-BF16 (p.ej. Tiger Lake)
    # bf16 se emula por software y sale perdiendo frente a fp32.
    ratio = gemm["bfloat16"] / gemm["float32"]
    mixed_worth_it = ratio > 1.1
    veredicto = "RECOMENDADA" if mixed_worth_it else "DESCARTADA (usar fp32)"
    print(f"\n  bf16/fp32 = {ratio:.2f}x -> precision mixta: {veredicto}")

    # Escalado por hilos: el SMT suele degradar GEMM porque los dos hilos logicos
    # compiten por la misma unidad vectorial.
    print(f"\n--- Escalado por hilos (fp32 {size}x{size}) ---")
    candidates = sorted({1, 2, physical, max_threads})
    scaling = {}
    best_threads, best_gflops = physical, 0.0
    for n in candidates:
        torch.set_num_threads(n)
        gflops = bench_gemm(size, torch.float32, max(3, iters // 2))
        scaling[str(n)] = round(gflops, 1)
        marker = ""
        if gflops > best_gflops:
            best_gflops, best_threads, marker = gflops, n, "  <-- mejor"
        print(f"  {n:2d} hilos {gflops:7.1f} GFLOPS{marker}")
    print(f"\n  Recomendado en config: num_threads = {best_threads}")

    torch.set_num_threads(original)
    return {
        "cpu": platform.processor(),
        "cores_physical": physical,
        "cores_logical": max_threads,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "gemm_gflops": gemm,
        "bf16_ratio": round(ratio, 3),
        "mixed_precision_recommended": mixed_worth_it,
        "thread_scaling_gflops": scaling,
        "recommended_threads": best_threads,
    }


def model_section(cfg, quick: bool) -> dict | None:
    """Throughput de un paso de entrenamiento real. Requiere la Fase 4."""
    try:
        from src.model.gpt import GPT
    except ImportError:
        print("\n=== MODELO ===")
        print("  src/model/gpt.py aun no existe (Fase 4).")
        print("  Vuelve a ejecutar este banco cuando el modelo este implementado.")
        return None

    m, t = cfg.model, cfg.training
    torch.set_num_threads(t.num_threads)
    model = GPT(m)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = m.vocab_size * m.d_model

    opt = torch.optim.AdamW(model.parameters(), lr=t.learning_rate,
                            betas=(t.beta1, t.beta2), weight_decay=t.weight_decay)
    batch = t.micro_batch_size
    x = torch.randint(0, m.vocab_size, (batch, m.block_size))
    y = torch.randint(0, m.vocab_size, (batch, m.block_size))

    def step():
        _, loss = model(x, targets=y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    seconds = _time_it(step, warmup=2, iters=2 if quick else 4)
    rss_peak = proc.memory_info().rss

    tok_s = batch * m.block_size / seconds
    tokens_day = tok_s * 86400 * THERMAL_DERATE
    total_tokens = t.max_steps * t.tokens_per_step(m.block_size)
    days = total_tokens / tokens_day

    print("\n=== MODELO ===")
    print(f"  Parametros       : {n_params / 1e6:.2f} M "
          f"({(n_params - n_embed) / 1e6:.2f} M sin embeddings)")
    print(f"  Micro-batch      : {batch} x {m.block_size} = {batch * m.block_size} tokens")
    print(f"  Paso fwd+bwd+opt : {seconds:.2f} s  ->  {tok_s:.0f} tokens/s")
    print(f"  RAM del proceso  : {rss_peak / 1e9:.2f} GB "
          f"(delta {(rss_peak - rss_before) / 1e6:.0f} MB)")
    print(f"\n  Sostenido (derate {THERMAL_DERATE}): {tokens_day / 1e6:.1f} M tokens/dia")
    print(f"  Presupuesto {total_tokens / 1e6:.0f} M tokens -> {days:.1f} dias  "
          f"({t.max_steps} pasos x {t.tokens_per_step(m.block_size)} tokens)")

    return {
        "params": n_params,
        "params_non_embedding": n_params - n_embed,
        "micro_batch_size": batch,
        "block_size": m.block_size,
        "seconds_per_step": round(seconds, 3),
        "tokens_per_second": round(tok_s, 1),
        "rss_peak_gb": round(rss_peak / 1e9, 2),
        "tokens_per_day_derated": round(tokens_day),
        "projected_days": round(days, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Banco de pruebas de rendimiento en CPU")
    ap.add_argument("--config", default=None, help="ruta al JSON de configuracion")
    ap.add_argument("--quick", action="store_true", help="menos iteraciones")
    ap.add_argument("--no-model", action="store_true", help="omitir el benchmark del modelo")
    ap.add_argument("--out", default=None,
                    help="destino del JSON (por defecto benchmarks/baseline.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Configuracion: {cfg.name}  <-  {cfg.source}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_name": cfg.name,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hardware": hardware_section(args.quick),
    }
    if not args.no_model:
        model = model_section(cfg, args.quick)
        if model:
            report["model"] = model

    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nLinea base guardada en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
