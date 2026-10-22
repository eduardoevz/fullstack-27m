# Fase 13 — Inferencia y producto (v2)

**Fecha:** 2026-10-22 al 2026-10-23
**Estado:** 🟡 Parcial. Demo con relleno en el cursor, README con v1 contra v2 y curvas del v2 están hechos; **la cuantización int8 se midió y no se entrega** (puerta de velocidad no cumplida).
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

Dejar el modelo v2 utilizable y presentable: cuantización dinámica int8 para CPU (se mide velocidad y degradación), demo con FIM («completar en el cursor»)
y README con la comparación v1 contra v2 y las curvas de ambas (`docs/plan-mejoras-modelo.md`, Fase 13).

## Qué se construyó

- **Demo con relleno en el cursor** (`src/cli.py`, `demo_v2.bat`). Hallazgo previo: `cli.py` declaraba `CURSOR_MARK` e importaba `frame_fim_prompt`, pero **nunca los usaba**; el
  modo FIM que documentaba no existía. Ahora `split_cursor(texto)` separa prefijo y sufijo (primer `<CURSOR>`; sufijo vacío admitido) y `complete(...)` arma el prompt FIM o el normal.
  En FIM no se vetan los marcadores (el relleno necesita cerrar con `<|endoftext|>`); sin cursor se vetan, como en la Fase 11. Se añaden `--rep-penalty` y `:rep`.
  `demo_v2.bat` carga `config/model_v2_long_c.json` y `hito-0018310.pt`. 5 tests nuevos (`tests/test_cli.py`).
- **`src/eval/quant_bench.py`:** mide tok/s con caché KV (prompt de 200, 100 tokens nuevos, 4 hilos, mediana de 5), tamaño del `state_dict` y el cociente frente a fp32.
- **Curvas del v2** (`src/eval/plot_curves.py` con `--csv` de varios tramos, `merge_runs` y 3 tests): el tramo posterior reemplaza los pasos que solapan y se quitan filas repetidas.
  La salida por defecto para v1 es byte a byte la de antes. Resultado: `docs/curvas-entrenamiento-v2.svg` (1.833 puntos de entrenamiento, 36 de validación, pasos 10–18.310).
- **README:** sección «v2 frente a v1», nota de estado, uso de `demo_v2.bat` y cifras actualizadas (suite: 322 tests).
- **`samples/demo-cli-v2.md`:** transcripción real de una sesión (una generación por bloque, sin selección), con sus fallos.

## Cuantización int8: resultado negativo

Puerta fijada **antes de medir**: se entrega solo si es ≥ 1,3× más rápida que fp32 (y, si lo fuera, val loss ≤ +0,02 y sintaxis estricta a 256 ≤ −5 puntos). Como la puerta
de velocidad falla, la degradación de calidad **no se midió**.

| Variante (PyTorch 2.14 CPU, motor `onednn`) | tok/s | Frente a fp32 | Tamaño |
|---|---|---|---|
| fp32 | 103,9 | 1,00× | 81,8 MB |
| int8 (todas las `nn.Linear`) | 83,6 | **0,80×** | 45,7 MB |
| int8 con `lm_head` en fp32 | 75,7 | 0,73× | 64,6 MB |

La cuantización dinámica reduce el archivo un 44 %, pero con un lote de un token por paso el sobrecoste de cuantizar las activaciones supera lo que se gana en la multiplicación
(causa **no verificada**: no se perfiló). El motor disponible es solo `onednn` y PyTorch avisa de que `torch.ao.quantization` y la creación de tensores cuantizados están
desaconsejados, así que no hay una vía estable que probar sin cambiar de herramienta. Datos: `benchmarks/v2_quant_bench.json`. Se queda fp32.

## Verificación

- Suite completa: **322 tests en verde** (314 antes de la fase + 5 de `test_cli.py` + 3 de `test_plot_curves.py`).
- Sesión real de `src/cli.py` con el modelo final: relleno en el cursor correcto en TS, y también una muestra donde el relleno de Python no cierra (agota los 200 tokens). Está en la transcripción.
- Curvas v1: regeneradas y comparadas con la versión anterior, idénticas.
- Cada cifra del README se contrastó con los JSON de `benchmarks/` y con `avance/fase-11-entrenamiento-fim.md`.

## Desviaciones y límites

- Sin cuantización entregada (resultado negativo, arriba).
- El README compara la pérdida de validación de v2 con reservas: el conjunto de validación de v2 es otro.
- La velocidad de v2 (104 tok/s) y la de v1 (90 tok/s, Fase 7) se midieron en momentos distintos; no se atribuye la diferencia al modelo.
- La transcripción es una sola muestra por bloque; no representa la tasa de aciertos (ver la evaluación de 200 casos de la Fase 11).
