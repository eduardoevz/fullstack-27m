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
| [04](fase-04-transformer.md) | Arquitectura del Transformer | ✅ Completa | `8b6ead2` |
| [05](fase-05-entrenamiento.md) | Bucle de entrenamiento optimizado para CPU | ✅ Completa | `1ad7bd6` |
| [06](fase-06-entrenamiento.md) | Entrenamiento (~8 días) | ✅ Completa (sin muestras intermedias) | `8cef616` |
| [07](fase-07-inferencia-evaluacion.md) | Inferencia, evaluación y demo | ✅ Completa (demo como transcripción) | `7438aa7` |
| [08](fase-08-diagnostico-decodificacion.md) | Mejora v1→v2: diagnóstico, evaluación y decodificación | ✅ Completa (puerta parcial) | `b62b96f` |
| [09](fase-09-datos.md) | Mejora v1→v2: datos (limpieza, fuente moderna, mezcla v2) | ✅ Completa (piso FastAPI sin cumplir; Next.js sin App Router) | `f1ce480` |
| [10](fase-10-fim.md) | Mejora v1→v2: formato FIM (prueba de 1.000 pasos) | 🟡 Parcial (código completo; puerta no cumplida: (b) 0,925 y (c) 5 % frente a 44,5 %) | `d608244` |

Leyenda: ✅ completa · 🟡 parcial · ⬜ pendiente

## Presupuesto del proyecto

| Métrica | Objetivo | Real |
|---|---|---|
| Parámetros | 20.45 M (antes 26.75 M; vocab 16k) | 20.45 M (analítico) |
| Tokens de entrenamiento | 500 M | 499,99 M en `train.bin` (de 822,6 M medidos) |
| Pasos de optimizador | 15.260 | 15.260 completados |
| Throughput sostenido | ~64 M tokens/día | 1.103 tok/s medios en los 15.260 pasos (Fase 06); antes: 1.133 tok/s con entrenamiento real de 200 pasos (Fase 05); 1.047–1.132 en el banco (≈ 63–68 M tokens/día con derate 0,7) |
| Duración del entrenamiento | ~7.8 días | ≈ 5,25 días de cómputo a 1.103 tok/s medios; ≈ 9 días de calendario (26-sep a 5-oct) en sesiones |
| Validez sintáctica (100 fragmentos) | — | 23 % estricta · 71 % recortada (archivos reales: 97 %) |
| Suite de 20 prompts | — | sintaxis 60 % estricta / 75 % recortada; en tema 60 % |
| Inferencia con caché KV | interactiva | 90 tok/s (10 sin caché) |
| Con `rep=1,1` (Fase 8): sintaxis estricta 256/512 tokens | — | 46 % / 63 % (antes 23 % / 33 %) |
| Pérdida de validación | — | 3,44 (paso 500) → **1,477** (mejor, paso 15.000) |

La columna «Real» se rellena a medida que las fases la vayan midiendo.
