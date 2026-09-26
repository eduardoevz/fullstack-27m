# Fase 05 — Bucle de entrenamiento optimizado para CPU

**Fecha:** 2026-09-26
**Estado:** ✅ Completa (con `torch.compile` sin probar: falta el compilador; ver decisiones abiertas)
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

Escribir `src/train.py`: AdamW, LR con calentamiento y coseno, acumulación de gradiente, checkpoints con reanudación
exacta y rotación, registro a CSV y guardia térmica. Medir el tok/s real con acumulación real.

## Qué se hizo

Módulos pequeños y testeables en `src/training/`, y la CLI en `src/train.py`:

- `schedule.py`: LR con calentamiento lineal de 500 pasos (pico 6e-4) y coseno hasta 6e-5.
- `data.py`: `BatchSampler` con offsets aleatorios sobre el memmap uint16, sin DataLoader; su estado (generador) se guarda.
- `optim.py`: AdamW (0,9; 0,95). Weight decay 0,1 **solo en matrices**; normas y embedding atado sin decay.
- `checkpoint.py`: guarda modelo, optimizador, paso, mejor val, estado del sampler y RNG (torch/numpy/python). Escritura atómica
  (`.tmp` + `os.replace`). Rotación: los 2 últimos `ckpt-*.pt` + `best.pt`.
- `loop.py`: `run_step` (4 micro-batches, pérdida/4, recorte 1,0) y `evaluate`.
- `train.py`: log a CSV (paso, pérdida, LR, tok/s, RAM, grad-norm, val), evaluación en un conjunto fijo de val, `Ctrl+C` →
  termina el paso, guarda y sale, y guardia térmica (aviso si la mediana móvil de tok/s cae bajo el 70 % de su máximo).
  Se niega a empezar si ya hay checkpoints y no se pasó `--resume`, para no pisarlos.
- `config/model_27m.json`: `checkpoint_interval` 500 → **50** (~25 min de entrenamiento).
- `tests/test_training.py`: 10 pruebas escritas antes del código. Total del repo: **63 en verde**.

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| LR: calentamiento, pico, mitad del coseno, suelo y monotonía | ✓ |
| Grupos de weight decay: sin duplicados, todos cubiertos, embedding y normas sin decay | ✓ |
| Sampler: forma, dtype, targets desplazados 1, determinista, estado restaurable | ✓ |
| Checkpoint: ida y vuelta idéntica, sin `.tmp` sobrantes, rotación correcta | ✓ |
| **Reanudación exacta (test):** 6 pasos seguidos = 3 + guardar + recargar en un modelo con otra semilla + 3; pérdidas y pesos idénticos bit a bit | ✓ |
| Acumulación de 4 micro-batches = un batch 4× (gradientes y pérdida) | ✓ |
| **Corrida real de 200 pasos** (config real, 32.768 tokens/paso) | pérdida 9,63 (paso 10) → **4,69** (paso 200); objetivo del plan: < 7 ✓ |
| **Corte y reanudación reales:** proceso muerto en el paso 120, reanudado del checkpoint del paso 100 | Pasos 110 y 120 idénticos a la primera corrida (pérdida 6,2491 y 5,9620; grad-norm 2,096 y 2,082). Sin salto |
| RAM durante la corrida | 1,55 → 1,71 GB, sin crecimiento sostenido |

## Rendimiento medido con acumulación real

- **Media 1.133 tok/s** (rango 1.042–1.162 entre ventanas de 10 pasos), en 20 ventanas. Coincide con el banco de la Fase 04
  (1.047–1.132): la sospecha de que el banco subestimaba el throughput real **no se confirma de forma apreciable**.
- Los últimos 20 pasos bajaron a ~1.045 tok/s; no sé si es calentamiento o ruido de la máquina. La guardia térmica no saltó.
- **Días para 500 M tokens:** 5,1 sin descuento; **7,3 con derate 0,7**, dentro del rango de 7,3–7,9 anotado en la Fase 04.
- Sigue siendo ~14 % por debajo del prototipo de 1.311 tok/s (causa sin identificar). No lo investigué más allá de esta medida.

## Límites y decisiones abiertas

1. **`torch.compile` no se pudo probar:** no hay `cl.exe` ni `triton` en la máquina. No instalé nada. Instalar VS Build Tools
   (~3 GB) es una decisión del Director; la ganancia en CPU Windows es incierta y no está medida. La bandera `--compile` existe
   pero no está probada.
2. **La evaluación en validación solo se ejercitó en una prueba corta** (config de humo: val 9,67 a los 4 pasos, y guardado de
   `best.pt`). En los 200 pasos reales no hubo evaluación porque `eval_interval` es 500. La ruta funciona, pero no la vi con
   entrenamiento largo.
3. **La reanudación real se probó con una muerte forzada entre checkpoints**, no con `Ctrl+C` sobre el proceso en marcha (la
   captura de `Ctrl+C` está implementada pero no la probé contra un entrenamiento real).
4. **Los checkpoints pesan ~245 MB** (modelo + estado de Adam), no ~430 MB como decía el plan. Con 2 últimos + best son ~0,75 GB.
5. Una nota de la corrida: en el CSV de prueba los pasos 110–120 aparecen dos veces (corrida original y reanudada) porque se
   añade al mismo archivo; es esperable al reanudar.
6. Siguen abiertas las del corpus (Next.js y FastAPI casi ausentes, archivos autogenerados, `.d.ts`) y el nombre `27m`.

## Siguiente

Fase 06 — Entrenamiento completo (~7–8 días) en sesiones reanudables. Antes de lanzarlo: desactivar suspensión, conectar el
cargador, pausar Windows Update y cerrar aplicaciones pesadas. No arranca sin aprobación del Director.
