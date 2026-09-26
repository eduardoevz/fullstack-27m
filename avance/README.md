# Registro de avance

Bitácora del proyecto, fase por fase. Cada archivo recoge qué se hizo, si la fase quedó
completa **según el plan de implementación**, las desviaciones respecto a lo planificado
y la verificación ejecutada con sus cifras.

El criterio es la honestidad, no el optimismo: una fase parcial se documenta como parcial.

El plan que se contrasta aquí es [`docs/plan-implementacion.md`](../docs/plan-implementacion.md).

## Estado

| Fase | Descripción | Estado | Commit |
|---|---|---|---|
| [00](fase-00-cimientos.md) | Cimientos y banco de pruebas | ✅ Completa | `11c5309` |
| [01](fase-01-dataset.md) | Dataset Full Stack (ingesta, filtrado, deduplicación) | ✅ Completa | `52220f0` |
| [02](fase-02-tokenizer.md) | Tokenizer BPE orientado a sintaxis de código | ✅ Completa | `d0017a6` |
| [03](fase-03-encoding.md) | Codificación a memmap binario | ✅ Completa | `d82d186` |
| 04 | Arquitectura del Transformer | ⬜ Pendiente | — |
| 05 | Bucle de entrenamiento optimizado para CPU | ⬜ Pendiente | — |
| 06 | Entrenamiento (~8 días) | ⬜ Pendiente | — |
| 07 | Inferencia, evaluación y demo | ⬜ Pendiente | — |

Leyenda: ✅ completa · 🟡 parcial · ⬜ pendiente

## Presupuesto del proyecto

| Métrica | Objetivo | Real |
|---|---|---|
| Parámetros | 20.45 M (antes 26.75 M; vocab 16k) | 20.45 M (analítico) |
| Tokens de entrenamiento | 500 M | 499,99 M en `train.bin` (de 822,6 M medidos) |
| Pasos de optimizador | 15.260 | — |
| Throughput sostenido | ~64 M tokens/día | — |
| Duración del entrenamiento | ~7.8 días | — |

La columna «Real» se rellena a medida que las fases la vayan midiendo.
