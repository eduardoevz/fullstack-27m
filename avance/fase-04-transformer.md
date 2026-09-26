# Fase 04 — Arquitectura del Transformer

**Fecha:** 2026-09-26
**Estado:** ✅ Completa (con una desviación de rendimiento que se decide en la Fase 05)
**Commit:** ver `avance/README.md`

## Objetivo según el plan

Implementar el Transformer estilo Llama (RMSNorm, RoPE, SwiGLU, sin sesgos, embeddings atados) con TDD, y medir por
primera vez su velocidad real, que hasta ahora venía de un prototipo desechable.

## Qué se hizo

- `src/model/rope.py`: tablas cos/sin (θ=10.000) con la convención `rotate_half`; `RotaryEmbedding` acepta un desfase de
  posición (lo usará la caché KV de la Fase 07) y extiende la tabla bajo demanda si T supera `block_size`.
- `src/model/gpt.py`: `RMSNorm` (cálculo en fp32), `CausalSelfAttention` (`qkv` fusionado + `F.scaled_dot_product_attention`
  causal), `SwiGLU`, `Block` pre-norm y `GPT` con cabeza atada al embedding. Inicialización `N(0, 0.02)` y, en las
  proyecciones residuales (`attn.proj`, `mlp.down`), `0,02/√(2·n_layer)`. Interfaz `logits, loss = model(idx, targets)`,
  la que ya esperaba `src/bench.py`.
- `tests/test_model.py`: 14 pruebas escritas antes del código. Total del repo: **53 en verde**.
- Sin `torch.compile` y sin precisión mixta (fp32 puro, decidido en la Fase 00).

## Verificación funcional

| Comprobación | Resultado |
|---|---|
| Nº de parámetros vs. `param_count()` | **20.453.760**, exacto (14,16 M sin embeddings) |
| Sin sesgos; `lm_head.weight is tok_emb.weight` | ✓ (el peso atado no se cuenta dos veces) |
| Causalidad (cambiar el token t no altera logits < t) | ✓ |
| Determinismo con semilla fija | ✓ |
| Pérdida inicial | 9,8165 en la corrida real; ln(16.384) = 9,704 (+0,11, dentro de lo previsto por el atado) |
| Gradientes finitos y no nulos en todos los parámetros | ✓ |
| RoPE: conserva la norma; q·k depende solo de la distancia relativa; desfase = corte de tabla; T > block_size | ✓ |
| SDPA causal = atención manual con máscara; RMSNorm = fórmula | ✓ |
| Sobreajuste de un batch, modelo diminuto (32 secuencias) | pérdida < 0,1 ✓ |
| **Sobreajuste de un batch con la config real** (4 × 128 tokens, AdamW lr 1e-3) | 9,82 → **0,0899 en 73 pasos** (32 s, RAM 0,91 GB) ✓ |

## Rendimiento medido (primera vez con el modelo real)

`python src/bench.py`, micro-batch 16×512, 4 hilos, fp32. La máquina es ruidosa: el GEMM fp32 dio entre 184 y 251 GFLOPS
entre corridas, y el throughput del 16k bajó en cada repetición (posible calentamiento).

| Vocabulario | Corridas (tok/s) | Días para 500 M tokens (con derate 0,7) | RAM |
|---|---|---|---|
| **16.384 (20,45 M par.)** | 1.132 · 1.083 · 1.047 | 7,3 · 7,6 · 7,9 | 1,27–1,38 GB |
| 32.768 (26,75 M par.) | 926 · 842 | 8,9 · 9,8 | 0,89 GB |

- **Razón de velocidad 16k/32k medida: 1,17 – 1,24.** La Fase 02 usó 1,24 (1.311/1.057). El 16k gana mientras la razón sea
  > 1,034 (=1/0,9673), así que **la elección de 16k se confirma con amplio margen**. Con la puntuación de la Fase 02 y la
  velocidad real: 16k ≈ 3,34 × 1.083 = 3.615 frente a 32k ≈ 3,45 × 926 = 3.196.
- **Velocidad real frente al prototipo: −14 % a −20 %** (1.047–1.132 frente a 1.311). Queda **fuera de la tolerancia de ±10 %**
  que fijó el plan. No sé la causa: el prototipo era otro código y no lo comparé línea a línea.
- **Consecuencia para el calendario:** con el derate de 0,7, entre **7,3 y 7,9 días** (el plan original de 27 M decía 7,8).
  Sin derate serían ~5,5 días. Como las propias mediciones ya arrastran calentamiento, aplicar además 0,7 puede contar dos
  veces; la cifra fiable saldrá de la Fase 05, con entrenamiento real.
- **Sesgo conocido del banco:** hace un `opt.step` por micro-batch, mientras el entrenamiento real lo hará una vez cada 4
  micro-batches. El tok/s real debería quedar algo por encima. No lo cuantifiqué.
- **RAM del 32k menor que la del 16k (0,89 vs 1,3 GB):** es contraintuitivo (sus logits son el doble) y no lo he explicado;
  puede deberse al orden de las corridas o al asignador. Ambas caben con holgura.

`benchmarks/baseline.json` guarda la primera corrida del 16k (1.132 tok/s), que es la mejor de las tres; el rango honesto es el de
la tabla.

## Desviaciones y decisiones abiertas

1. **Velocidad por debajo de la tolerancia** (arriba). No se cambia nada ahora: el presupuesto de 500 M tokens sigue cabiendo en
   ~7,3–7,9 días, similar al plan original. Se mide de nuevo con entrenamiento real en la Fase 05, y solo entonces se decide si
   recortar tokens o probar palancas (hilos, tamaño de micro-batch, `torch.compile` con puerta de decisión).
2. **Nombre `27m`** sigue en repo y config aunque el modelo tiene 20,45 M parámetros (ver Fase 02).

## Para la Fase 05 (anotado)

- Bajar `checkpoint_interval` de 500 a ~50 pasos (~30 min de entrenamiento) y guardar un checkpoint al recibir `Ctrl+C`.
- Grupos de weight decay (solo matrices, no normas ni embeddings) y medir el throughput con acumulación de gradiente real.

## Siguiente

Fase 05 — Bucle de entrenamiento optimizado para CPU: AdamW, LR con calentamiento y coseno, acumulación de gradiente,
checkpoints con reanudación exacta, registro a CSV y guardia térmica.
