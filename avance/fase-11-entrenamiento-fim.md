# Fase 11 — Entrenamiento largo con FIM (v2) y decodificación

**Fecha:** 2026-10-09 al 2026-10-22
**Estado:** 🟡 Parcial. El entrenamiento de 18.310 pasos terminó y el modelo supera a v1 en el relleno FIM, pero **no alcanza a v1 en validez sintáctica** (55 % contra 63 % estricta, 57 % contra 68 % recortada).
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

Entrenar v2 durante todo el presupuesto (18.310 pasos de optimizador, lote 16×4, contexto 512) desde la prueba de 1.000 pasos de la Fase 10
(`init_from` = `v2-trial-1000.pt`), con la mitad de las filas en formato FIM, y medir si el relleno mejora a v1 sin perder la generación normal.

## Qué se construyó

- **Vigilancia de FIM durante el entrenamiento** (`src/eval/fim_monitor.py`): cada 500 pasos mide, con 60 casos, la pérdida FIM del medio
  (frente a la referencia de v1, 2,5955) y la **tasa de parada** (el relleno termina antes de agotar el presupuesto). Aborta si, desde
  `fim_abort_step`, la razón supera `fim_abort_ratio` o la tasa de parada cae bajo `fim_abort_stop_rate`.
- **FIM por líneas** (`fim_line_rate` 0,5), **peso extra al cierre del medio** (`fim_stop_weight`), **FIM a nivel de texto** y
  **diagnóstico de frontera** (`src/eval/fim_boundary.py`), que separa la pérdida del contenido de la del cierre.
- **Hitos** (`milestone_interval` 1.000): copia permanente del checkpoint en `checkpoints_*_hitos`, porque `ckpt-*.pt` rota (se guardan 2).
- **`src/eval/medicion_500.py`:** espera el paso múltiplo de 500, conserva su checkpoint, lo evalúa con 200 casos y resume la salud del tramo.
- **Veto de marcadores FIM al generar** (`banned_ids` en `generate`, `fim_banned_ids(tok)`): se activa en la CLI, el arnés y las tareas, y
  se desactiva con `--suffix`. Test nuevo; suite: **314 tests en verde**.
- Lanzadores propios por corrida: `entrenar_v2_long[_b|_c].bat` y `estado_v2_long[_b|_c].bat`.

## El entrenamiento, en cuatro tramos

Todos los tramos son un solo entrenamiento (LR 3e-4 → 3e-5 en coseno, 18.310 pasos); cada reanudación partió del checkpoint indicado.
Los cambios de regla se decidieron **al ver cada aborto**, no antes de medir.

| Tramo | Pasos | Qué ocurrió | Cambio al reanudar |
|---|---|---|---|
| Original | 0 → 9.519 | Aborto en el paso 3.000 (la regla no se cumplió); se movió el chequeo a los pasos 6.000 y luego 7.500. Aborto en el 9.000 (0,860 × v1) y en el 9.500 (0,899 × v1) | Peso de cierre 20 → 5; límite de razón 0,85 → 0,88 |
| b | 9.400 → 10.000 | Se reanudó desde el 9.400 con peso de parada 5. Se detuvo en el 10.000 con tasa de parada 28,3 % (< 0,4) | Evaluación de 200 casos del 10.000; peso de parada 5 → 8 |
| c | 10.000 → 13.885 | Aborto en el 10.500 (parada 31,7 %). Evaluación de 200 casos: el peso 8 no mejoraba el parseo (24 %) | Umbral de parada 0,4 → 0,25 |
| c (cont.) | 13.885 → 18.310 | El monitor de 60 casos oscila entre 25 % y 48 % sin tendencia (un caso = 1,7 puntos); parada ordenada en el 13.885 | Umbral de parada 0,25 → 0,2 |

Las paradas ordenadas (13.885 y 14.521) no fueron abortos de la regla: guardaron checkpoint y se reanudó con el `.bat`. La tasa de parada cayó a 28 % tras subir el peso
y se recuperó a 38–48 % a partir del paso 11.500; no tengo medido qué parte se debe al peso y cuál al ruido del monitor.

### Salud medida cada 500 pasos (monitor de 60 casos)

| Paso | Val loss | Val FIM | FIM medio (× v1) | Tasa de parada |
|---|---|---|---|---|
| 10.000 | 1,4071 | 1,4776 | 2,107 (0,812) | 28,3 % |
| 12.000 | 1,3882 | 1,4602 | 2,187 (0,843) | 38,3 % |
| 14.000 | 1,3763 | 1,4459 | 2,210 (0,852) | 40,0 % |
| 16.000 | 1,3660 | 1,4353 | 2,228 (0,858) | 45,0 % |
| 18.000 | 1,3599 | 1,4299 | 2,127 (0,819) | 41,7 % |

El rendimiento fue de ~1.000–1.200 tok/s con 1,5–2,8 GB de RAM. El paso 18.310 no tiene fila de validación propia; el 18.000 es el último medido.

## Evaluación del relleno FIM (200 casos, frente a v1)

| Paso | Pérdida FIM (× v1) | Similitud | Parsea | Exacto |
|---|---|---|---|---|
| 9.000 | 0,866 | 0,279 | 30,5 % | 3,0 % |
| 10.000 | 0,834 | 0,277 | 23,5 % | 1,5 % |
| 10.500 | 0,865 | 0,268 | 24,0 % | 2,0 % |
| 11.000 | 0,849 | 0,279 | 29,0 % | 1,0 % |
| 12.000 | 0,829 | 0,296 | 31,0 % | 2,5 % |
| 14.000 | 0,853 | 0,306 | 29,5 % | 3,0 % |
| **18.310 (final)** | **0,834** | **0,302** | **35,5 %** | **4,5 %** |
| v1 | 1,000 | 0,257 | 44,5 % | 2,0 % |

Frontera en el final: 0,836 × v1 por texto y 0,726 por token (el mejor de la serie). El parseo cayó a ~24 % al pasar del 9.000 al 10.500 y se recuperó
hasta 35,5 %, la mayor parte en el tramo final de bajada del LR. Sigue 9 puntos por debajo de v1.

## Generación normal: suite de 20 prompts (100 muestras, 200 tokens, conjunto de prueba)

Hallazgo: el v2 **escribía marcadores FIM** (`<|fim_suffix|>`, `<|fim_middle|>`) dentro de la generación normal: 19/100 muestras con T 0,7 y 46/100 con
penalización 1,1 (v1: 0). Entre las muestras que terminaban, parseaba el 65 % (v1: 93 %). Se veta ese token al muestrear.

| Modelo | Decodificación | Estricta | Recortada | En tema |
|---|---|---|---|---|
| v1 | T 0,7 / pen. 1,0 | 46 % | 66 % | 69 % |
| v1 | T 0,8 / pen. 1,1 | 63 % | 68 % | 69 % |
| v2 sin veto | T 0,7 / pen. 1,0 | 36 % | 51 % | 87 % |
| v2 sin veto | T 0,8 / pen. 1,1 | 37 % | 42 % | 82 % |
| v2 con veto | T 0,7 / pen. 1,0 | 44 % | 59 % | 88 % |
| v2 con veto | T 0,8 / pen. 1,1 | 46 % | 55 % | 82 % |
| **v2 con veto y ajuste** | **T 0,8 / pen. 1,2** | **55 %** (IC95 42–67) | **57 %** | **79 %** |

### Ajuste de decodificación (conjunto `dev`, 45 fragmentos × 256 tokens y 2 muestras por prompt)

Mismo método y reglas que la Fase 8 (ganancia mínima 0,05 de objetivo; «en tema» no cae más de 5 puntos). Solo ayudó la penalización por
repetición: la sintaxis estricta pasó de 24 % (base) a 38 / 42 / 47 % con 1,05 / 1,1 / 1,2. Los n-gramas (27–31 %), min-p (36–40 %, con 27 % de
fragmentos repetitivos) y bajar la temperatura (16–20 %) no mejoraron. La combinación 1,2 + n-gramas 3 quedó en 42 %. Se eligió **penalización 1,2**
y se validó una sola vez en el conjunto de prueba, congelado.

## Resultado frente a lo esperado

- **Cumple:** el relleno FIM supera a v1 en pérdida (0,834 ×), similitud (0,302 contra 0,257) y coincidencia exacta (4,5 % contra 2,0 %); el «en tema» de la suite
  también (79 % contra 69 %); y hay menos repetición (0,04 contra 0,09).
- **No cumple:** la validez sintáctica de v1 con su mejor decodificación (63 % estricta, 68 % recortada) y el parseo FIM de v1 (44,5 %).
- **Causa de la brecha sintáctica:** no probada. Mi hipótesis es que dedicar la mitad del entrenamiento a FIM resta calidad a la generación normal
  (se ven repeticiones de decoradores y variables duplicadas), y que un peso de cierre alto empuja a terminar pronto. Ningún experimento lo confirma.

## Pendiente

- Decidir si hace falta un **ajuste fino corto con 0 % de FIM** para recuperar el modo normal; habría que medir primero cuánto FIM se pierde.
- Medir la sintaxis en 256/512 tokens y la suite de tareas (`run_tasks`) con el v2 final, como en la Fase 8.
- El paso 18.310 no tiene validación propia en el CSV; el checkpoint final está en `checkpoints_v2_long_c_hitos/hito-0018310.pt`.

## Archivos de datos

`benchmarks/fim_eval_*.json` (200 casos por paso), `benchmarks/v2_final_suite_test*.json`, `benchmarks/v2_decoding_search.json`.
Los CSV de entrenamiento están en `logs/` (no versionados).
