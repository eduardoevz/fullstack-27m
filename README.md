# fullstack-27m

Un modelo de lenguaje causal de **26.75 M de parámetros**, pre-entrenado **desde cero
y enteramente en CPU**, especializado en autocompletado y generación de código Full Stack:
React, Next.js y TypeScript en el frontend; Node (Express, NestJS) y Python (FastAPI, Flask)
en el backend.

No hay GPU en ninguna parte de este proyecto. Todo el pipeline —datos, tokenizer,
arquitectura, entrenamiento e inferencia— está diseñado alrededor de las restricciones
de un portátil: un Intel i5-1135G7 de 4 núcleos y 16 GB de RAM.

> **Estado:** Fase 0 completada (cimientos y línea base). En construcción.

---

## Por qué 27M y no 50M

El objetivo inicial era un modelo de 50M de parámetros. Medir el hardware antes de
escribir el modelo cambió la decisión.

Un modelo de 50M parámetros necesita ~1.000M de tokens para acercarse al óptimo de
Chinchilla. A los 303 tokens/s que esta CPU sostiene con esa arquitectura, son **~45 días**
de entrenamiento continuo. Con 27M parámetros el throughput sube a 1.057 tokens/s y el
presupuesto de 8 días permite 500M de tokens: **19 tokens por parámetro**, prácticamente
el óptimo (534M).

Mismo tiempo de cómputo. Un modelo considerablemente mejor.

| Configuración | Parámetros | tok/s | Tokens en 8 días | Tokens/parámetro |
|---|---|---|---|---|
| d=512, L=10 | 48.3 M | 303 | 150 M | 3.1 (infraentrenado) |
| **d=384, L=8** | **26.7 M** | **1057** | **500 M** | **19 (casi óptimo)** |

## La precisión mixta no siempre acelera

El plan original contemplaba bf16. La medición lo descartó:

```
GEMM 1024x1024, 4 hilos
  float32     213.8 GFLOPS
  bfloat16     59.1 GFLOPS     <-- 3.6x mas lento
```

El i5-1135G7 (Tiger Lake) tiene AVX-512 pero **no** AVX512-BF16 ni AMX. Sin instrucciones
nativas, bf16 se emula en software y cada operación cuesta más que en fp32. El
entrenamiento usa fp32 puro.

## El SMT perjudica al entrenamiento

Los 8 hilos lógicos de la CPU no son 8 unidades de cómputo. Los dos hilos de cada núcleo
comparten una sola unidad vectorial:

```
 1 hilos   105.7 GFLOPS
 2 hilos   214.4 GFLOPS
 4 hilos   349.8 GFLOPS     <-- mejor (= nucleos fisicos)
 8 hilos   228.2 GFLOPS     <-- 35% peor que 4
```

`num_threads = 4`, verificado por medición.

---

## Arquitectura

Estilo Llama en lugar de GPT-2: a igualdad de parámetros converge mejor, con coste de
cómputo equivalente.

```
d_model      384          n_layers   8        n_heads   6  (head_dim 64)
contexto     512          vocab      16.384   FFN       1024 (SwiGLU)
Norm         RMSNorm      Posicion   RoPE     Bias      ninguno
Embeddings   tied (entrada y salida comparten pesos)

Parametros   20.45 M  (14.2 M sin contar embeddings)
```

Un detalle que condiciona el diseño: con `d_model=384` y un vocabulario de 32.768, la
proyección final consume **~46% del cómputo por token**. El tamaño del vocabulario no es
una decisión cosmética, y se tomó midiendo compresión real contra velocidad (Fase 2):
el vocabulario de **16.384** comprime el 96,7 % de lo que comprime el de 32.768 (umbral: 81 %)
y el modelo corre más rápido, así que ganó. Consecuencia: el modelo pasa de 26.75 M a
**20.45 M parámetros** (el nombre `27m` del repositorio ya no es exacto; se conserva por
estabilidad de rutas).

## Presupuesto de entrenamiento

| Métrica | Valor |
|---|---|
| Tokens | 500 M |
| Pasos de optimizador | 15.260 |
| Batch efectivo | 32.768 tokens (micro-batch 16 × acumulación 4) |
| Throughput sostenido | ~64 M tokens/día (derate térmico 0.7) |
| Duración estimada | ~7.8 días |

---

## Estructura

```
config/model_27m.json    Fuente de verdad: arquitectura, entrenamiento, datos, rutas
src/config.py            Carga y valida el config; conteo analitico de parametros
src/bench.py             Banco de pruebas: GFLOPS, escalado por hilos, tok/s
src/data/                Ingesta en streaming, filtrado de dominio, deduplicacion
src/tokenizer/           BPE byte-level entrenado sobre el corpus propio
src/model/               RMSNorm, RoPE, SwiGLU, GPT
src/train.py             Bucle de entrenamiento con acumulacion y reanudacion exacta
src/eval/                Validez sintactica y suite de prompts del dominio
tests/                   Causalidad, conteo de parametros, round-trip del tokenizer
benchmarks/baseline.json Linea base reproducible del hardware
```

Los datos y los checkpoints viven en `C:/llm-fullstack-data/`, **fuera de OneDrive**:
un checkpoint de ~430 MB reescrito cada 500 pasos durante ocho días es una invitación
a que el cliente de sincronización bloquee un archivo a mitad de escritura.

## Uso

```bash
python src/bench.py                  # linea base de rendimiento
pytest tests/ -v                     # suite de tests
```

## Requisitos

Python 3.14, PyTorch 2.14 CPU. Ver `requirements.txt`.

---

## Qué es y qué no es este proyecto

Un modelo de 27M parámetros entrenado con 500M de tokens genera código
**sintácticamente correcto y estructuralmente reconocible**: imports plausibles, JSX
bien formado, decoradores de FastAPI en su sitio. No genera código listo para producción,
y no compite con asistentes comerciales entrenados con órdenes de magnitud más de cómputo.

El objeto de este repositorio es el pipeline de ingeniería y las decisiones medidas que
lo sostienen.
