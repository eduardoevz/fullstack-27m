# Plan de mejora del modelo: de v1 a v2

> **Estado:** aprobado por el Director el 2026-10-06. **Fase 8 completada** (ver `avance/fase-08-diagnostico-decodificacion.md` y la sección «Actualización tras la Fase 8»); las fases 9–13 siguen pendientes y cada una requiere aprobación antes de empezar. Continúa la numeración del plan original (fases 0–7, ya cerradas).

## Contexto

La v1 de `fullstack-27m` (20,45 M de parámetros, 500 M de tokens, val 1,477) cerró las 8 fases del plan original. Sus
resultados medidos son modestos: **23 %** de archivos que parsean enteros (71 % recortando la cola), solo **14 de 100**
archivos cerrados por el propio modelo, **14 %** de líneas repetidas (reales: 2 %), 60 % «en tema» en la suite, y casi
nada de Next.js (0,14 % de los tokens) y FastAPI (0,02 %). Este documento propone cómo mejorarlo **con medición en cada
paso**, dentro de las restricciones del proyecto (CPU, una laptop, ~95 M tokens/día de cómputo).

El objetivo no es prometer una cifra sino ordenar las palancas por **impacto esperado / coste** y poner una puerta de
decisión entre fases para no gastar días de cómputo sin evidencia.

## Diagnóstico: qué falla y por qué (con evidencia y con hipótesis)

| Síntoma medido en v1 | Causa probable | Estado |
|---|---|---|
| Repite líneas y bloques (14 % vs 2 %) | Muestreo sin penalización de repetición + modelo pequeño | Hipótesis fuerte, barata de probar |
| Cierra pocos archivos (14/100 en 256 tokens) | El muestreo de entrenamiento coge ventanas de 512 tokens en offsets aleatorios del flujo; los documentos miden de media ~1.265 tokens, así que casi nunca se ve un archivo completo ni su inicio `<\|file\|><\|lang\|>` en la posición 0 | **Hipótesis a confirmar** (diagnóstico de la Fase 8) |
| Next.js/FastAPI casi ausentes | La fuente (`github-code-clean`) es antigua; el rebalanceo no crea datos que no existen (Fase 3) | Medido |
| Ruido en el corpus | `.d.ts`, plantillas `<%= %>`, Python 2, cabeceras de licencia (Fases 1 y 7) | Medido |
| Se pierde a larga distancia | Contexto de 512 tokens y modelo pequeño | Esperable |
| No sirve para autocompletar *en medio* de un archivo | Solo se entrenó a continuar el texto (izquierda → derecha) | Falta de diseño |
| Métrica poco discriminante | n = 20 prompts, 1 muestra por prompt, solo sintaxis; los resultados no se pueden comparar con rigor | Medido (Fase 7) |
| ~24 tokens/parámetro | Los modelos pequeños mejoran bastante más allá del óptimo Chinchilla; v1 está lejos de saturar | Conocimiento general; se mide en la Fase 11 |

## Actualización tras la Fase 8 (2026-10-06)

Medido, no supuesto (cifras y reservas en la bitácora de la fase):

- **El diagnóstico descartó en buena parte la hipótesis del cierre de archivos.** Al inicio de un archivo la pérdida es *menor* que en
  una ventana al azar, y al final real el modelo asigna una P(`<|endoftext|>`) mediana de 0,38 (≈ 0 en mitad del archivo). Que terminaran
  pocos fragmentos se debía sobre todo al presupuesto de 256 tokens: la mediana de un archivo real mide 520.
- **La decodificación arregló los bucles:** con `repetition_penalty = 1,1` los fragmentos repetitivos bajan de 16 % a 1 % (256 tokens) y de
  52 % a 7 % (512), la sintaxis estricta pasa de 23 % a 46 % y de 33 % a 63 %, y terminan 36 y 63 de 100 en vez de 14 y 26.
- **No arregló la capacidad:** la sintaxis recortada no mejora, el pass@1 funcional sigue en 1,3 % y el Python sin nombres sin definir
  empeora (100 % → 57 % y 90 % → 47 %, con muestras pequeñas).

**Ajustes recomendados al resto del plan** (a decidir por el Director antes de cada fase):

1. **Fase 9 (datos) sube de prioridad:** es la palanca que puede mover la capacidad, y la más probable causa del 1,3 % funcional.
2. **Fase 10:** el muestreo alineado a documentos pierde justificación como arreglo del cierre de archivos; **FIM se mantiene** (autocompletar
   en medio del código). Los tokens FIM siguen requiriendo ampliar el vocabulario.
3. **Antes de la Fase 11:** añadir una métrica funcional más sensible (el pass@1 con 30 tareas está en el suelo y no distingue nada), p. ej.
   acierto de la línea siguiente sobre código real de validación y pérdida por dominio.
4. **Las metas de «terminados» y «repetición» de la tabla de éxito** (≥ 50 % y ≤ 5 %) ya están casi alcanzadas solo con decodificación a
   256 tokens; las metas que quedan realmente por mover son sintaxis recortada, funcional y «en tema».

## Principios

1. **Congelar v1** como línea base (`best.pt` copiado a `v1-best.pt`, métricas en `benchmarks/`). Toda mejora se
   compara contra ella con **las mismas semillas y los mismos validadores**.
2. **Un cambio por vez cuando sea barato probarlo** (decodificación, datos). Cuando no (días de cómputo), agrupar y
   reconocer que el efecto queda confundido.
3. **Separar conjuntos:** un *dev set* de prompts nuevos para ajustar, y la suite actual de 20 como *test set* que no se
   toca al ajustar (evita sobreajustar la métrica).
4. **Puerta de decisión tras cada fase** con criterio escrito antes de medir. Cada fase: aprobación del Director,
   bitácora en `avance/`, commit + push.

## Fases propuestas (en el orden recomendado)

### Fase 8 — Diagnóstico, evaluación rigurosa y decodificación (sin reentrenar) · ~1–2 días de trabajo, ~horas de cómputo

Es la más rentable: no cuesta días de entrenamiento y decide si las siguientes se justifican.

- **Diagnóstico de v1:** pérdida por posición dentro de la ventana; pérdida en ventanas que empiezan en un documento
  vs. ventanas al azar; probabilidad que el modelo asigna a `<|endoftext|>` al final de archivos reales. Confirma o
  descarta la hipótesis de los inicios y finales de documento.
- **Decodificación:** penalización de repetición, `no_repeat_ngram`, `min_p`, y rejilla de temperatura/top-p, ajustadas en
  el *dev set* y evaluadas una vez en el *test set*. Se añaden a `src/sample.py` (con tests, como hasta ahora).
- **Evaluación mejorada:** 5 muestras por prompt (n = 100 en la suite) con intervalos de confianza por bootstrap;
  presupuesto de 512 tokens además de 256; **pass@1 funcional** con ~30 tareas pequeñas de Python/JS con tests
  unitarios; comprobación de nombres sin definir en Python (`pyflakes`) y, opcionalmente, `tsc --noEmit` en TS.
- **Entregable:** `benchmarks/v1_baseline_eval.json`, informe de qué parte de los defectos ya se arregla solo con decodificar.
- **Puerta:** si la repetición baja a ≤ 5 % y la estricta sube de forma clara, se sabe cuánto de «mala calidad» era
  decodificación. Si no, las fases 9–11 están justificadas.

### Fase 9 — Datos: más Next.js/FastAPI y menos ruido · ~2–3 días de trabajo + descarga

- **Primero lo gratis:** hay 822,6 M tokens codificados y solo se usaron 500 M. Los ~322 M restantes se pueden reutilizar
  sin descargar nada (con los mismos filtros nuevos).
- **Limpieza** (reglas deterministas, con tests como las de `src/data/filters.py`): excluir `.d.ts`, plantillas con
  `<%`/`{{`, Python 2 (`print` sin paréntesis), cabeceras de licencia, archivos con errores obvios.
- **Filtro de calidad con el propio modelo v1:** puntuar la perplejidad por documento y descartar la cola peor (basura,
  texto no-código). Barato y específico de este corpus.
- **Fuente moderna** para lo que falta (Next.js App Router, FastAPI, NestJS, TSX). **Decisión del Director:** las fuentes
  recientes (p. ej. The Stack v2/StarCoderData) suelen exigir aceptar términos en Hugging Face. Antes de comprometerse se
  verificará disponibilidad y licencia con una descarga pequeña de prueba. Meta de mezcla: Next.js ≥ 5 %, FastAPI ≥ 3 %,
  tsx ≥ 15 % de los tokens (hoy 0,14 %, 0,02 % y 4,1 %).
- **Conjunto de validación por dominio** (Next.js, FastAPI, NestJS, React) para medir la pérdida donde importa.
- **Puerta:** ≥ 50 M tokens de Next.js + FastAPI de calidad verificada a mano en una muestra de 50 archivos. Si no se
  consigue, la mejora de esos frameworks se declara fuera de alcance y se documenta.

### Fase 10 — Formato de entrenamiento: archivos completos, cierre de archivo y FIM · ~2 días de trabajo, ~horas de pruebas

- **Muestreo alineado a documentos:** una fracción de las ventanas (p. ej. 50 %) empieza en `<|file|>`, y los documentos
  cortos entran enteros, para que el modelo aprenda a abrir y **cerrar** archivos.
- **Fill-in-the-Middle (FIM)** (formato prefijo–sufijo–medio, en una fracción de los documentos): es lo que permite
  autocompletar *en medio* del código, que es lo que hace un asistente real. Requiere 3–4 tokens especiales nuevos:
  se añaden filas al embedding (inicializadas con la media) en vez de reentrenar el tokenizer; el conteo de parámetros
  cambia en ~0,001 M y hay que actualizar `ModelConfig.param_count()` y sus tests.
- **Cambios con TDD:** `src/training/data.py` (sampler), `src/data/encode.py` (índice de documentos para alinear),
  `src/sample.py` y `src/cli.py` (modo FIM).
- **Puerta:** prueba corta (≈ 1.000 pasos, ~8 h) partiendo de v1: la pérdida en ventanas de inicio de documento baja y
  la probabilidad de `<|endoftext|>` sube, sin empeorar la validación general más de un margen fijado de antemano.

### Fase 11 — Entrenamiento continuo v2 · ~6–7 días de cómputo (≈ 600 M tokens)

- Partir de los pesos de v1 y **seguir entrenando** con la mezcla nueva (Fase 9) y el formato nuevo (Fase 10):
  ≈ 600 M tokens más (≈ 54 tokens/parámetro acumulados). Cada 500 M tokens son ~5,25 días.
- **LR:** recalentar con warmup corto (200 pasos) a un pico menor que el original (~3e-4) y coseno hasta 3e-5; optimizador
  reiniciado. Riesgo conocido: un pico demasiado alto «olvida» parte de v1; se vigila con la validación vieja.
- Mismo contexto de 512. Mismas herramientas de sesiones (`entrenar.bat`, `parar.bat`, `estado.bat`), checkpoints,
  parada limpia y revisión cada 1–2 días.
- **Muestras cada 1.000 pasos** en `samples/` (pendiente de la Fase 6: se implementa en `train.py` con test, esta vez sí).
- **Puerta:** la validación (vieja y por dominio) mejora; si se estanca 2.000 pasos, se corta y se pasa a la Fase 12.
- **Expectativa honesta:** la ganancia de pérdida de validación será modesta (centésimas a pocas décimas); lo que
  más debería moverse son los defectos de formato (cierre, repetición) y los frameworks nuevos. No se promete una cifra.

### Fase 12 — Recocido de dominio y contexto 1.024 (opcional) · ~1–2 días de cómputo

- ~80–100 M tokens de la mejor calidad (Next.js, FastAPI, NestJS, TSX) con el LR decayendo hasta ~0: suele dar un
  empujón desproporcionado en el dominio objetivo.
- **Contexto 1.024** por ajuste fino con el mismo RoPE (con interpolación de posiciones si hace falta). Coste de cómputo
  estimado ~+13 % por token; se mide antes de decidir. Se hace solo si la Fase 8 muestra que el contexto limita.

### Fase 13 — Inferencia y producto (opcional)

- Cuantización dinámica int8 de las capas lineales para CPU (se mide velocidad y degradación; bf16 sigue descartado).
- Demo con FIM («completar en el cursor»), README con la comparación v1 vs. v2 y curvas de ambas.

## Lo que se descarta (con motivo)

- **Más parámetros (40–50 M):** ~45 días de cómputo para un óptimo Chinchilla de 50 M (medido en la Fase 0): no cabe.
- **Cambios de arquitectura (GQA, QK-norm…):** a 20 M de parámetros y 512 de contexto el cuello de botella son datos y
  formato, no la atención; el beneficio esperado es pequeño y reentrenar desde cero cuesta ~5 días por prueba.
- **bf16 / mixed precision:** 3,6× más lento en este hardware (medido).
- **GPU en la nube:** acabaría con la identidad «entrenado en CPU local». Si el Director quisiera un modelo mayor, sería
  un proyecto aparte, no una mejora de este.
- **Datos sintéticos de un LLM grande (destilación):** posible, pero cambia la naturaleza del proyecto («desde cero con
  datos propios») y tiene implicaciones de licencia; solo con decisión explícita del Director.
- **`torch.compile`:** sigue sin probarse (falta `cl.exe`); instalar VS Build Tools (~3 GB) solo si una medición pequeña
  mostrara ganancia. No es prioritario.

## Calendario y presupuesto

| Fase | Trabajo | Cómputo | Puerta |
|---|---|---|---|
| 8 Diagnóstico + decodificación + evaluación | 1–2 días | horas | ¿cuánto se arregla sin reentrenar? |
| 9 Datos | 2–3 días | descarga + codificación (~horas) | ≥ 50 M tokens de Next.js/FastAPI verificados |
| 10 Formato + FIM | 2 días | ~8 h de prueba | mejora en inicio/cierre de documento |
| 11 Entrenamiento continuo | revisiones | **~6–7 días** | val vieja y por dominio mejoran |
| 12 Recocido + contexto 1.024 | 1 día | 1–2 días | suite y dominio |
| 13 Inferencia y producto | 1–2 días | horas | velocidad/calidad medidas |

Total aproximado: **2–3 semanas de calendario**, de las cuales ~8–10 días son entrenamiento en sesiones reanudables.
Se recomienda empezar solo con la Fase 8: es barata y su resultado decide si el resto vale la pena.

## Objetivos de éxito (metas, no promesas)

Medidos con los mismos validadores, 100 fragmentos, semillas fijas y el *test set* intacto:

| Métrica | v1 | Meta v2 |
|---|---|---|
| Sintaxis estricta (256 tokens) | 23 % | ≥ 50 % |
| Archivos cerrados por el propio modelo (256 tokens) | 14 % | ≥ 50 % |
| Líneas repetidas por fragmento | 14 % | ≤ 5 % |
| Suite de 20 prompts «en tema» (con 5 muestras) | 60 % | ≥ 75 % |
| Next.js + FastAPI en tema (subconjunto de la suite) | por medir en la Fase 8 | mejora respecto a v1 con IC |
| pass@1 funcional (tareas con tests) | por medir en la Fase 8 | mejora respecto a v1 con IC |

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Las fuentes modernas exigen términos/token de Hugging Face | Prueba pequeña de descarga antes de comprometer; plan B: solo limpieza + datos existentes |
| Recalentar el LR degrada v1 | Pico menor, warmup corto, vigilancia con la validación vieja; v1 se conserva |
| Ajustar a la suite de 20 prompts la «sobreajusta» | Dev set separado; test set intacto hasta el final |
| FIM empeora la continuación normal | Fracción moderada de FIM (p. ej. 50 %) y medición en ambos modos |
| El entrenamiento de ~7 días se interrumpe | Mismo sistema reanudable y probado de la Fase 6 |
| Las ganancias son menores de lo esperado | Puertas escritas de antemano; cada fase se documenta aunque sea negativa |

## Archivos que se tocarían (a modo de guía)

`src/sample.py`, `src/eval/*` (evaluación y tareas funcionales), `src/data/filters.py` y `src/data/ingest.py`,
`src/data/encode.py`, `src/training/data.py`, `src/train.py` (muestras cada 1.000 pasos, FIM), `src/config.py` (+ tests),
`src/model/gpt.py` (solo si cambia el vocabulario), `config/` (config v2) y `docs/` + `avance/` (bitácora de cada fase).
Se reutilizan: `forward_cached`/`generate`, el validador sintáctico, el sistema de checkpoints y reanudación, y los `.bat`
de sesión.

## Verificación

Cada fase termina con: tests nuevos en verde + suite completa; cifras en `benchmarks/` comparadas con v1; muestras en
`samples/`; `avance/fase-NN-*.md` con desviaciones y honestidad sobre lo que no se logró; commit y push.
Al cerrar todo: tabla v1 vs. v2 con las seis métricas de arriba y las curvas de ambas.

## Decisiones que necesito del Director

1. ¿Se aprueba empezar **solo con la Fase 8** (recomendado) y decidir el resto con sus resultados?
2. ¿Se permite explorar fuentes de datos modernas que pidan aceptar términos en Hugging Face (Fase 9)?
3. ¿Se acepta que v2 amplíe el vocabulario con tokens FIM (Fase 10)?
