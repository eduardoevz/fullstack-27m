# Fase 08 — Diagnóstico, evaluación rigurosa y decodificación (sin reentrenar)

**Fecha:** 2026-10-06
**Estado:** ✅ Completa según el plan de la fase. **La puerta se cumple solo en parte** (ver «Lectura de la puerta»)
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

`docs/plan-mejoras-modelo.md` (Fase 8): sin tocar el modelo, (1) confirmar o descartar por qué v1 no cierra archivos, (2) medir cuánto de
la mala calidad se arregla solo con la decodificación y (3) construir una evaluación más rigurosa (conjuntos dev/test, varias muestras
por prompt con intervalos de confianza, pass@1 funcional, nombres sin definir). Todo con reglas escritas **antes** de medir.

## Qué se hizo

- **v1 congelado** en `C:/llm-fullstack-data/checkpoints/`: `v1-best.pt` (paso 15.000, SHA-256 `fc340ad2…9986`, idéntico a `best.pt`) y
  `v1-final.pt` (paso 15.260, SHA-256 `82600968…5b19`). Todas las cifras de v1 salen de `v1-best.pt` con semillas fijas.
- **`src/eval/diagnose.py`:** pérdida por posición, inicio de documento frente a ventana al azar, P(`<|endoftext|>`) en el final real y
  estadística del muestreo de entrenamiento → `benchmarks/v1_diagnosis.json`.
- **Decodificación (`src/sample.py`):** `repetition_penalty` (solo sobre los últimos 64 tokens generados, no el prompt),
  `no_repeat_ngram_size` y `min_p`, todos apagados por defecto (un test comprueba que sin activarlos la salida es idéntica a v1).
- **Evaluación:** `src/eval/stats.py` (bootstrap IC95 % y `pass_at_k`), `src/eval/harness.py` (generación, puntuación y resumen
  reutilizables), `src/eval/prompts.py` con un conjunto **DEV** de 20 prompts nuevos (los 20 originales quedan como **TEST** congelado),
  `run_suite.py` con `--set`/`--samples` e IC, `syntax_check.py` con parámetros de decodificación y a 256/512 tokens, métrica de nombres
  sin definir con `pyflakes`, y `src/eval/tasks.py` + `run_tasks.py` con **30 tareas funcionales** (20 Python, 10 JS) con tests unitarios.
- **`src/eval/tune_decoding.py`:** ajuste sobre DEV con la regla fijada de antemano (ver abajo).
- **Tests:** de 96 a **219 en verde** (cada grupo escrito antes del código: generación, estadísticas, diagnóstico, harness, tareas con
  filtros de seguridad, selección del ajuste).

## 1. Diagnóstico de v1 (sin reentrenar)

Sobre `val.bin` (repos disjuntos de train). Regla fijada antes de medir: hipótesis «el muestreo por offsets al azar impidió aprender a
abrir/cerrar archivos» = *confirmada* si la mediana de P(eot) < 0,3 y la pérdida de los primeros 64 tokens en inicios de documento es
≥ 10 % mayor que al azar; *descartada* si la mediana de P(eot) ≥ 0,5; *mixta* en otro caso.

| Medida | Resultado |
|---|---|
| Pérdida de los primeros 64 tokens: ventana que empieza en `<\|file\|>` / ventana al azar | 1,42 / 2,20 (cociente **0,64**) |
| P(`<\|endoftext\|>`) al final real de 300 archivos de ≤ 500 tokens | mediana **0,38**; > 0,5 en el 46 %; > 0,1 en el 73 % |
| P(eot) media en mitad de un archivo | 0,0008 |
| Longitud de los documentos (tokens) | mediana 520; p25 241; p75 1.139; 50 % mide más que la ventana de 512; solo 27,6 % mide ≤ 256 |
| Ventanas de entrenamiento al azar que contienen un eot / que empiezan en un inicio de documento | 33,6 % / 0,087 % |

**Veredicto por la regla fijada: «mixta».** La lectura, que es más interesante que la etiqueta:
- **Abrir archivos no es un problema:** la pérdida al inicio de un archivo es *menor* que en una ventana al azar. (Mi comparación estaba
  confundida: una ventana al azar arranca sin contexto y por eso es más difícil; no lo vi al fijar la regla.)
- **Cerrar archivos tampoco es un fallo grave:** el modelo asigna al final real una P(eot) mediana de 0,38 y está calibrado (≈ 0 en mitad
  del archivo). **Que solo 14 de 100 fragmentos terminaran en la Fase 7 se debe en gran parte al presupuesto de 256 tokens:** la mediana
  de un archivo real mide 520 tokens y solo el 27,6 % cabe en 256. Es, en buena medida, un artefacto de la medición.
- Consecuencia para el plan: **pierde fuerza el muestreo alineado a documentos de la Fase 10** como remedio; FIM se justifica por otras
  razones (autocompletar en medio del código), no por esta.

## 2. Línea base de v1 (decodificación original: T=0,8, top-k 40, top-p 0,95)

Control: la corrida de 100 fragmentos a 256 tokens **reproduce exactamente** las cifras de la Fase 7 (23 % / 71 % / 14 terminados /
86 % / 14 % repetidas / 16 % repetitivos), lo que confirma que el refactor no cambió nada.

| Medida | 256 tokens | 512 tokens |
|---|---|---|
| Sintaxis estricta (100 fragmentos) | 23 % | 33 % |
| Sintaxis recortada | 71 % | 73 % |
| Terminados por el modelo (y % de ellos que parsea) | 14 (86 %) | 26 (88 %) |
| Líneas repetidas (media) / fragmentos con ≥ 30 % repetido | 14 % / 16 % | **30 % / 52 %** |
| Python que parsea y no tiene nombres sin definir | 100 % (de 6) | 90 % (de 10) |
| Archivos reales de validación (calibración, estricta) | 97 % | 96 % |

Más largo = más repetición: a 512 tokens más de la mitad de los fragmentos son bucles.

Suite de prompts (5 muestras por prompt; IC95 % por bootstrap sobre prompts):

| Conjunto | Estricta | Recortada | En tema |
|---|---|---|---|
| TEST, T=0,7 | 46 % (33–59) | 66 % (53–78) | 69 % (53–83) |
| TEST, T=0,8 | 53 % (41–64) | 65 % (53–77) | 67 % (51–81) |
| DEV, T=0,7 | 54 % (43–64) | 79 % (67–89) | 79 % (67–90) |

pass@1 funcional (30 tareas × 5 muestras, T=0,2): **1,3 %** (IC95 0–4 %); Python 0 %, JS 4 %; 1 tarea de 30 resuelta alguna vez.

## 3. Ajuste de la decodificación sobre DEV

Regla fijada antes de medir: objetivo = (estricta + (1 − fragmentos repetitivos)) / 2 con 45 fragmentos a 256 tokens; admisible si «en
tema» (DEV) no cae más de 5 puntos y la repetición media > 0; se elige la mejor solo si supera a la base en ≥ 0,05 de objetivo.

| Configuración | Objetivo | Estricta | Repetitivos | En tema (dev) |
|---|---|---|---|---|
| base | 0,567 | 27 % | 13 % | 85 % |
| **rep=1,1** | **0,733** | 49 % | 2 % | 88 % |
| rep=1,05 | 0,678 | 40 % | 4 % | 88 % |
| rep=1,2 | 0,700 | 40 % | 0 % | 85 % |
| ngram=3 / 4 / 6 | 0,644 / 0,678 / 0,700 | 29 / 36 / 40 % | 0 % | 90 / 85 / 85 % |
| min_p=0,05 / 0,1 | 0,500 / 0,533 | 31 / 38 % | 31 % | 85 / 82 % |
| temp=0,6 / 0,7 | 0,544 / 0,544 | 27 % | 18 % | 80 / 82 % |
| rep=1,1 + ngram=6 | 0,722 | 44 % | 0 % | 88 % |

**Decisión por la regla: `repetition_penalty = 1,1`.** `min_p` empeora la repetición (31 % de fragmentos repetitivos; la causa probable es que estrecha la distribución, pero no la medí);
bajar la temperatura no ayuda; combinar con n-gramas no supera a la penalización sola. Registro completo en `benchmarks/decoding_search.json`.

## 4. Evaluación final sobre TEST (una sola vez, configuración elegida)

| Medida | v1 base | v1 + rep 1,1 |
|---|---|---|
| Sintaxis estricta, 256 tokens | 23 % | **46 %** |
| Sintaxis estricta, 512 tokens | 33 % | **63 %** |
| Sintaxis recortada, 256 / 512 | 71 % / 73 % | 68 % / 77 % |
| Terminados por el modelo, 256 / 512 | 14 / 26 | **36 / 63** |
| Líneas repetidas (media), 256 / 512 | 14 % / 30 % | **6 % / 11 %** |
| Fragmentos con ≥ 30 % repetido, 256 / 512 | 16 % / 52 % | **1 % / 7 %** |
| Python que parsea y no tiene nombres sin definir, 256 / 512 | 100 % (6) / 90 % (10) | **57 % (14) / 47 % (17)** |
| Suite TEST, estricta (T=0,8, ×5) | 53 % (41–64) | 63 % (52–74) |
| Suite TEST, recortada | 65 % (53–77) | 68 % (55–80) |
| Suite TEST, en tema | 67 % (51–81) | 69 % (54–82) |
| pass@1 funcional | 1,3 % (0–4) | 1,3 % (0–4) |

Diferencias pareadas en la suite (mismos prompts, IC95 % bootstrap): estricta **+10 puntos (−3 a +23)**, recortada +3 (−8 a +14), en tema
+2 (−6 a +10). **Ninguna es estadísticamente distinguible de cero con 20 prompts.** En fragmentos desde cero, la diferencia de sintaxis
estricta es de +23 puntos a 256 tokens y +30 a 512 (aproximación normal, IC95 % ≈ ±13 puntos): esa sí es clara.

## Lectura de la puerta (criterio del plan)

> «Si la repetición baja a ≤ 5 % y la estricta sube de forma clara, se sabe cuánto de la mala calidad era decodificación. Si no, las
> Fases 9–11 están justificadas.»

- **Lo que sí se arregló solo con decodificar:** los bucles (fragmentos repetitivos 16 % → 1 % a 256 y 52 % → 7 % a 512) y, con ello, la
  sintaxis estricta (se duplica) y el cierre de archivos (14 → 36 y 26 → 63 terminados). La repetición media baja a 6 % (256) y 11 %
  (512): **cerca del 5 % pero no por debajo**, así que el umbral exacto no se cumple.
- **Lo que no se arregló:** la sintaxis «recortada» no mejora (71 → 68; 73 → 77: ruido), el pass@1 funcional sigue en el suelo (1,3 %) y
  **la limpieza de nombres en Python empeora** (100 % → 57 % y 90 % → 47 %): con la penalización el modelo parsea más, pero inventa más
  identificadores. Es cautela, no conclusión: los fragmentos Python que parseaban antes eran pocos (6 y 10) y probablemente triviales.
- **Conclusión:** una parte grande del defecto *de superficie* era decodificación (y medición). **La capacidad de fondo no cambia:** v1
  no sigue instrucciones de docstring ni escribe funciones correctas. Las Fases 9–11 (datos, formato, más entrenamiento) siguen
  justificadas para eso, pero con una motivación distinta de la del plan original.

## Desviaciones y honestidad

1. **Muestras DEV del ajuste: ×2, no ×1** como decía el plan (para reducir ruido; ~2,5 min por configuración, total ~30 min).
2. **Se ajustó a 256 tokens**, pero la repetición empeora mucho a 512. La elección se confirmó después a 512, donde el efecto es mayor;
   aun así una penalización distinta podría ser mejor para generaciones largas. No se exploró.
3. **`tsc --noEmit` quedó fuera**, como se anticipó (sin tipos de react/next los errores de módulos faltantes son ruido).
4. **pass@1 funcional con efecto suelo:** con 1,3 % no distingue configuraciones; no pudo pesar en la decisión. Un juez funcional útil
   para este modelo necesita tareas mucho más fáciles o una métrica más fina (p. ej. acierto de la línea siguiente). Queda abierto.
5. **La regla de la hipótesis tenía un defecto de diseño** (comparaba contra ventanas sin contexto); se informó la etiqueta fijada
   («mixta») y la lectura corregida arriba.
6. **El conjunto DEV lo escribí yo después de conocer los prompts de TEST**; son del mismo dominio pero otros casos. No es independencia
   total. Además la suite TEST de la Fase 7 ya había sido vista, así que el «test congelado» lo es de aquí en adelante.
7. **Dos temperaturas en la línea base de la suite** (0,7 y 0,8): se repitió a 0,8 para que la comparación con la configuración elegida
   (T=0,8) sea pareada. La configuración elegida se evaluó una sola vez en TEST.
8. **Ejecución de código generado:** 2 de 150 finalizaciones (línea base) y 3 de 150 (con rep 1,1) fueron rechazadas por el filtro de
   seguridad y no se ejecutaron. Los filtros son mitigación, no un sandbox perfecto (ver `src/eval/tasks.py`).
9. **Predeterminados cambiados:** `src/sample.py` (CLI) y `src/cli.py` (demo) usan ahora `rep=1,1`; la función `generate` conserva el
   comportamiento de v1 por defecto. El README principal sigue citando las cifras de v1 con la decodificación original.
10. **Caveats estadísticos:** 100 fragmentos por medida y 20 prompts por suite; los IC son anchos y solo se afirma lo que el IC respalda.

## Siguiente

Recomendación (sin ejecutar nada sin aprobación): **mantener `rep=1,1` como decodificación por defecto de v1** y **revisar el plan de
mejora**: (a) bajar la prioridad del muestreo alineado a documentos de la Fase 10 y conservar FIM; (b) subir la de la Fase 9 (datos), que
es lo que puede mover la capacidad; (c) añadir una métrica funcional más sensible antes de la Fase 11. Ver `docs/plan-mejoras-modelo.md`.
