# Fase 10 — Formato de entrenamiento: FIM (autocompletar en medio del código)

**Fecha:** 2026-10-08 al 2026-10-09
**Estado:** 🟡 Parcial. Todo el código está hecho y probado, pero **la puerta de la prueba de 1.000 pasos NO se cumple**: pasa (a) y fallan (b) y (c).
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

Enseñar al modelo a completar *en medio* de un archivo (prefijo + sufijo → medio) con el formato FIM, y comprobarlo con una prueba corta de
1.000 pasos desde los pesos de v1 (decisión del Director: 1.000 pasos y margen +0,05). La Fase 11 (entrenamiento largo) no se ejecuta aquí.

## Qué se construyó

- **Tokens nuevos** `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` al final del vocabulario (ids 16384–16386, vocab 16387): ningún id
  existente ni `train.bin` cambia (`src/tokenizer/extend.py` → `tokenizer-v2.json`). Parámetros: 20.454.912.
- **Formato PSM:** `<|file|> <|lang_x|> <|fim_prefix|> PREFIJO <|fim_suffix|> SUFIJO <|fim_middle|> MEDIO <|endoftext|>`, sin máscara de pérdida.
- **`FimBatchSampler`** (`src/training/fim.py`): 50 % de las filas FIM (documento elegido en proporción a su longitud, dos cortes aleatorios,
  relleno con los documentos siguientes del flujo) y 50 % de ventanas planas. Con `fim_rate=0` es idéntico a `BatchSampler`.
- **Carga de pesos de v1** (`src/training/init_from.py`): copia los pesos, las filas nuevas del embedding = media de las existentes, optimizador y
  paso desde cero. `train.py` gana `--init-from`; `config.py` gana `fim_rate` e `init_from` con valores por defecto.
- **Inferencia:** `frame_fim_prompt` y `--suffix` en `sample.py`; marcador `<CURSOR>` en `cli.py`.
- **Evaluación** (`src/eval/fim_eval.py`) y lanzadores `entrenar_v2.bat` / `estado_v2.bat` con `config/model_v2.json`.
- Suite: **297 tests en verde**.

## La prueba (1.000 pasos, desde `v1-best.pt`)

LR pico 3e-4 → 3e-5 (coseno), warmup 100, lote 16×4, contexto 512, `fim_rate` 0,5, datos `tokens_v2`. Duración ≈ 1 día de calendario (el
rendimiento cayó a ~690 tok/s unas horas, con avisos de throttling, y se recuperó).

| Paso | Pérdida plana (val) | Pérdida FIM (val) |
|---|---|---|
| 0 (v1 + 3 filas nuevas) | 1,4804 | 1,8123 |
| 250 | 1,5032 | 1,6182 |
| 500 | 1,4843 | 1,5777 |
| 750 | 1,4581 | 1,5483 |
| 1.000 | 1,4451 | 1,5349 |

## Puerta (fijada antes de medir) y resultado

| Condición | Resultado | Límite | Estado |
|---|---|---|---|
| (a) Pérdida plana en `tokens_v2/val.bin` | 1,4451 | ≤ 1,5128 + 0,05 = 1,5628 | **Cumple** |
| (b) Pérdida del medio, FIM (prefijo+sufijo) / v1 (solo prefijo) | 2,328 / 2,516 = **0,925** | ≤ 0,85 | **Falla** |
| (c) % de reconstrucciones que parsean | FIM **5 %** · v1 **44,5 %** | FIM ≥ v1 | **Falla** |

Evaluación de infilling: 200 casos (documentos de validación que parsean, medio de 1–3 líneas, `fim_eval.json`). Exacto: FIM 0 % · v1 2 %;
similitud: FIM 0,214 · v1 0,257.

**Matiz sobre (c) y análisis posterior.** En la medición oficial recorté la salida de v1 al número de líneas del medio verdadero (v1 no sabe parar)
pero no la de FIM, que debía parar sola con `<|endoftext|>`. El modelo FIM **no aprendió a terminar el medio**: sigue generando (repite imports,
abre bloques nuevos). Para separar «no para» de «el contenido es malo», repetí la evaluación recortando también la salida de FIM
(`--trim-fim`, `fim_eval_trim.json`; análisis posterior, **no** cambia el veredicto de la puerta):

| | FIM recortado | v1 |
|---|---|---|
| Reconstrucciones que parsean | 18,5 % | 44,5 % |
| Similitud del medio | 0,178 | 0,257 |

Aun recortando, FIM queda por debajo de v1. Es decir, no es solo un problema de parada: en 1.000 pasos el modelo mejoró la predicción del medio
un 7,5 % (teacher-forcing) pero **no es todavía un buen rellenador**.

## Lectura honesta

- FIM **aprende**: la pérdida FIM bajó de 1,81 a 1,53 sin parar de mejorar, y la pérdida plana terminó mejor que v1 en este conjunto. No hay
  evidencia de un fallo de código (los tests pasan, el comportamiento es estable).
- Es compatible con la hipótesis del plan: 1.000 pasos (16 M de tokens, la mitad FIM) con el LR bajando a 3e-5 son pocos para aprender a *cerrar*
  el medio. Es una hipótesis, **no medida**; la Fase 11 la pondrá a prueba con muchos más pasos.
- El baseline de v1 con prefijo solo y salida recortada a las líneas del medio es una referencia favorable (conoce el número de líneas); sirve
  como cota, no como comparación en igualdad de condiciones.

## Desviaciones y limitaciones

- **Muestreo alineado a documentos para las filas planas: descartado** (la Fase 8 mostró que abrir/cerrar archivos no era el problema).
- **Cortes a nivel de token:** pueden partir un token BPE; limitación documentada, no corregida.
- **`domain_loss` no se ejecutó** sobre el modelo v2: la puerta (a) se midió con la validación general de `tokens_v2`, como indica el plan.
- No se generó demo en `samples/`: con este resultado habría sido poco representativa.
- Pérdida de validación medida en 6–`eval_batches` lotes fijos, no en todo `val.bin`.

## Pendiente para la Fase 11

Decidir (con el Director) la duración del entrenamiento largo y si se ajustan `fim_rate`, el LR pico o el peso de `<|endoftext|>` tras el medio.
Recordatorio: revocar/rotar el token de Hugging Face expuesto en la Fase 9.

## Ajustes posteriores al cierre (preparación de la Fase 11)

Tras el resultado de la puerta, con aprobación del Director, se prepararon tres cosas (sin lanzar la Fase 11):

1. **Cortes FIM por línea.** Hipótesis (sin medir): entrenamiento con cortes aleatorios a mitad de línea frente a evaluación con medios de líneas
   completas. `FimBatchSampler` gana `line_rate`/`newline_ids`: esa fracción de filas FIM corta justo después de un token con salto de línea.
   Con `line_rate=0` el comportamiento anterior es idéntico (test). Config de la Fase 11: `fim_line_rate` 0,5.
2. **Vigilancia con aborto temprano** (`src/eval/fim_monitor.py`): cada 500 pasos mide, en 60 casos fijos de líneas completas, la pérdida del
   medio (frente a la de v1 solo con prefijo, referencia fija 2,5955 en `benchmarks/fim_monitor_ref.json`) y la tasa de parada con
   `<|endoftext|>`. Regla escrita antes de entrenar: desde el paso 3.000 (≈ 98 M tokens) se aborta si la razón > 0,85 **o** la parada < 50 %.
   Al abortar se guarda un checkpoint y se sale. Probado en humo (20 pasos): aborta correctamente.
3. **Sin segunda prueba de 8 h:** el aborto temprano da la misma información.

`config/model_v2_long.json` (18.310 pasos ≈ 600 M tokens, LR 3e-4 → 3e-5, parte de `v2-trial-1000.pt`), `entrenar_v2_long.bat` y
`estado_v2_long.bat` quedan listos. Dato del humo: con los pesos de la prueba, la tasa de parada es 0 % y la razón de pérdida ya ronda 0,85
en casos de líneas completas (8 casos; no concluyente).
