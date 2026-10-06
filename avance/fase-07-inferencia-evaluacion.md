# Fase 07 — Inferencia, evaluación y demo

**Fecha:** 2026-10-06
**Estado:** ✅ Completa según el plan, con una desviación (el vídeo de la demo es una transcripción) y los resultados del modelo son modestos y están documentados tal cual
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

`src/sample.py` (temperatura, top-k, top-p, caché KV), `src/eval/syntax_check.py` (generar 100 fragmentos y medir qué
porcentaje parsea), `src/eval/prompts.py` (suite de 20 prompts del dominio), `src/cli.py` (demo de autocompletado) y el
README final con arquitectura, decisiones medidas, curvas, resultados y limitaciones.

## Qué se hizo

- **Caché KV** en el modelo (`forward_cached` en `src/model/gpt.py`). El `forward` de entrenamiento no se tocó, así que
  el modelo entrenado es el mismo. Se reutilizó el `offset` que `RotaryEmbedding` ya tenía previsto.
- **`src/sample.py`:** muestreo con temperatura (0 = determinista), top-k y top-p; para al emitir `<|endoftext|>`; si el
  contexto llena las 512 posiciones, descarta la caché y reprocesa la mitad final de la ventana; callback por token para
  mostrar texto en vivo. Enmarca el prompt como en el entrenamiento: `<|file|> <|lang_x|> texto`.
- **`src/eval/syntax.py` + `check_syntax.js`:** validador. Python con `ast.parse`; JS/JSX/TS/TSX con el parser de
  TypeScript 5.9 (vía node). Mide solo sintaxis. Incluye recorte al último punto válido, reparto 34/33/33 y resumen.
- **`src/eval/syntax_check.py`:** genera 100 fragmentos desde cero y los mide; incluye una **calibración con 100 archivos
  reales de validación** para comprobar que el validador no rechaza código bueno.
- **`src/eval/prompts.py` + `run_suite.py`:** 20 prompts (React, Next.js App Router, NestJS, Express, FastAPI, Flask,
  SQLAlchemy, pytest) y su ejecutor; guarda `samples/suite-20-prompts.md` y `benchmarks/prompt_suite.json`.
- **`src/cli.py` + `demo.bat`:** demo interactiva (escribes código, terminas con una línea `.`; comandos `:lang`, `:temp`,
  `:tokens`, `:quit`) con salida en vivo. Transcripción real en `samples/demo-cli.md`.
- **`src/eval/plot_curves.py`:** curvas de entrenamiento en SVG, sin añadir matplotlib.
- **README final** reescrito con resultados, limitaciones y uso.
- **Tests:** `test_generation.py` (16) y `test_syntax_eval.py` (14), escritos antes del código. Total del repo: **96 en verde**.

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| Caché KV = forward completo (prefill, token a token y tramo tras prefijo) | ✓ (tolerancia 1e-3 sobre logits de magnitud ~20; ver nota) |
| Generación determinista con caché = sin caché, token a token | ✓ |
| Velocidad con modelo real (100 tokens, prompt de 200, 4 hilos) | **90 tok/s con caché vs 10 tok/s sin ella** (~9×) |
| Validador: acepta TS con tipos, JSX y TSX; rechaza código roto | ✓ |
| **Calibración:** 100 archivos reales de validación (≤ 256 tokens) | 97 % parsea; los 3 fallos son una plantilla con `<%= %>`, un `print` de Python 2 y un archivo con un typo |
| **Validez sintáctica, 100 fragmentos generados** (paso 15.000, T=0,8, 256 tokens) | **23 % estricta · 71 % recortada** |
| Fragmentos que el modelo cerró solo (`<|endoftext|>`) | 14 / 100; de ellos parsean 12 (86 %) |
| Por lenguaje (estricta / recortada) | JS 24 % / 62 % · TS 27 % / 73 % · Python 18 % / 79 % |
| Repetición de líneas dentro del fragmento | 14 % (reales: 2 %); 16 % de los fragmentos con ≥ 30 % repetido (reales: 0 %) |
| **Suite de 20 prompts** | sintaxis estricta 12/20 (60 %) · recortada 15/20 (75 %) · «en tema» 12/20 (60 %) |
| Reproducibilidad | la medición de 100 fragmentos se corrió dos veces con las mismas semillas: cifras idénticas |
| Suite completa del repo | 96 pruebas en verde |

## Desviaciones y honestidad

1. **«Grabar la demo CLI»:** el plan pedía grabarla. Hay una **transcripción** de una sesión real (`samples/demo-cli.md`),
   no un vídeo. Grabar pantalla es una acción manual de Eduardo con `demo.bat`.
2. **La métrica estricta es dura y la recortada es una definición mía.** El límite de 256 tokens corta casi todos los
   fragmentos a mitad de una sentencia (solo 14 de 100 terminaron), así que «parsea el fragmento completo» castiga el
   límite de longitud más que la escritura. La métrica «recortada» (quitar las líneas finales cortadas y exigir
   ≥ 80 % de retención) la definí yo; el umbral del 80 % es arbitrario. Se informan **ambas** y la calibración con
   archivos reales; ninguna sustituye a la otra.
3. **Solo sintaxis.** Una propiedad duplicada en una `interface` o una variable sin declarar pasan el validador (son
   errores semánticos). «Parsea» no significa «funciona».
4. **«En tema» es una heurística de texto** (si aparece alguna de 3–4 cadenas esperadas), no una evaluación de
   corrección. Con 20 prompts y una generación por prompt, cada uno pesa 5 puntos porcentuales: no hay intervalo de
   confianza serio.
5. **Con 14 fragmentos terminados**, el 86 % de validez de esos es una cifra de muestra pequeña.
6. **Tolerancia en los tests de caché:** inicialmente fijé 1e-4 y el test falló por 1 en la cuarta cifra decimal con logits
   de magnitud ~20 (error de coma flotante, no de la caché: con 1e-3 pasan los tres casos). Es muy inferior a cualquier
   fallo real de posicionamiento (que cambiaría los logits en órdenes de magnitud más).
7. **TypeScript 7 no sirve** para el validador (es el port nativo y no expone la API de JavaScript); se instaló la rama 5.x
   en `C:/llm-fullstack-data/tools` (fuera del repo, decisión aprobada por el Director). `node --check` por sí solo no
   habría cubierto TS/JSX/TSX.
8. **Calidad del modelo:** estructura reconocible pero repite bloques, se pierde a larga distancia y rara vez cierra
   archivos; un prompt de FastAPI derivó a Flask. Coherente con 20,45 M de parámetros y con que Next.js y FastAPI casi
   no están en el corpus. No se hicieron mejoras al muestreo (p. ej. penalización de repetición) para no cambiar lo que
   se mide.
9. **Muestras de la Fase 6:** `samples/` ya no está vacía, pero contiene solo muestras del modelo final; las intermedias
   de la Fase 6 siguen sin existir.
10. **`plot_curves.py` y el SVG** se generaron y su contenido numérico es el del CSV, pero no revisé el dibujo en un visor.

## Cierre del proyecto

Con esta fase quedan completas las ocho (00–07). Abiertas, heredadas de fases anteriores: el desequilibrio del corpus
(Next.js, FastAPI), archivos autogenerados y `.d.ts`, el nombre `27m`, `torch.compile` sin probar y las muestras
intermedias de entrenamiento que no se guardaron.
