# Fase 00 — Cimientos y banco de pruebas

**Fecha:** 2026-09-23
**Estado:** ✅ Completa
**Commit:** `11c5309`

## Objetivo según el plan

Estructura del repositorio, dependencias y el arnés de benchmark que sirve de línea base
para todas las decisiones posteriores.

## Qué se hizo

### Estructura y configuración

- Árbol de directorios: `config/`, `src/{data,tokenizer,model,eval}/`, `tests/`,
  `benchmarks/`, `samples/`, `logs/`.
- `config/model_27m.json` como fuente de verdad única: arquitectura, entrenamiento,
  filtros de datos y rutas. Ningún hiperparámetro se escribe a mano fuera de aquí.
- `src/config.py`: carga el JSON en dataclasses inmutables y **falla en el arranque** si
  falta una clave, sobra una clave desconocida, `d_model` no es divisible entre `n_head`
  o `head_dim` es impar (RoPE lo exige par). Incluye `param_count()`, el conteo analítico
  contra el que la Fase 04 validará el modelo real.
- `.gitignore`, `README.md` y `requirements.txt`.

### Banco de pruebas

`src/bench.py` mide GFLOPS en fp32 y bf16, el escalado por número de hilos, y —cuando el
modelo exista— el throughput de un paso real de entrenamiento con el pico de RAM.
Vuelca todo a `benchmarks/baseline.json`.

### Dependencias

Instaladas: `tokenizers 0.23.2`, `datasets 5.0.1`, `pyarrow 25.0.1`, `psutil 7.2.2`,
`tqdm 4.70.1`, `pytest 9.1.1`. Ya estaban en el sistema: `torch 2.14.0+cpu`, `numpy 2.5.2`.
`pip check` sin conflictos.

### Repositorio

`git init`, commit inicial y publicación en https://github.com/eduardoevz/fullstack-27m
(público, rama `main`).

## Mediciones

```
CPU        : Intel i5-1135G7, 4 nucleos fisicos / 8 logicos
RAM        : 16.9 GB
Vectorizado: AVX512, MKLDNN activo

GEMM 1024x1024, 4 hilos
  float32     213.8 GFLOPS
  bfloat16     59.1 GFLOPS      bf16/fp32 = 0.28x

Escalado por hilos (fp32)
   1 hilo    105.7 GFLOPS
   2 hilos   214.4 GFLOPS
   4 hilos   349.8 GFLOPS   <-- mejor
   8 hilos   228.2 GFLOPS
```

### Decisiones que fijan estas cifras

**fp32, no bf16.** Tiger Lake tiene AVX-512 pero no AVX512-BF16 ni AMX. Sin instrucciones
nativas, bf16 se emula en software y queda 3.6x más lento. La precisión mixta —que el plan
original contemplaba— se descarta.

**`num_threads = 4`, no 8.** Los dos hilos lógicos de cada núcleo comparten una sola unidad
vectorial, así que usar los 8 lógicos degrada GEMM un 35% frente a usar los 4 físicos. Sobre
un entrenamiento de 8 días, esa única línea de configuración vale ~2.8 días.

Nota sobre fiabilidad: el GEMM fp32 a 4 hilos dio 213.8 GFLOPS en una sección y 349.8 en otra
dentro de la misma ejecución, por turbo boost y estado térmico. Las comparaciones válidas son
las internas a cada bloque (fp32 frente a bf16; hilos entre sí), que es donde se tomaron las
decisiones. Los valores absolutos tienen un margen amplio.

## Desviaciones respecto al plan

**1. Rutas absolutas en lugar de junctions.** El plan proponía enlazar `data/` y
`checkpoints/` dentro del proyecto con `mklink /J`. En su lugar, el config apunta directo a
`C:/llm-fullstack-data/`. Motivo: evita crear puntos de reparse dentro de una carpeta
sincronizada por OneDrive. Mismo resultado, menos superficie de fallo.

**2. El benchmark del modelo queda pendiente.** `bench.py` detecta que `src/model/gpt.py`
todavía no existe y omite esa sección. Los 1.057 tokens/s que sostienen el presupuesto de
8 días provienen de un prototipo desechable, **no** del modelo del repositorio. Se validarán
en la Fase 04. Si no se confirman, hay que revisar el presupuesto de tokens.

**3. Riesgo retirado.** El plan preveía que `datasets` podría no tener wheel para Python 3.14
y reservaba un fallback de descarga directa de parquet. Instaló limpio con wheels nativos
cp314, así que ese fallback no se implementa.

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| `python src/bench.py` | Ejecuta y escribe `benchmarks/baseline.json` |
| Conteo analítico de parámetros | 26.75 M — coincide con el objetivo |
| Presupuesto de tokens | 15.260 pasos × 32.768 = 500.04 M |
| Importación conjunta de dependencias | Sin conflictos; `pip check` limpio |
| Autoría del commit | Eduardo Velasquez, sin atribución a IA |
| `.gitignore` | Se corrigió `data/` → `/data/`: el patrón sin anclar también ignoraba `src/data/`, que habría quedado fuera del repositorio sin avisar |

## Siguiente

Fase 01 — Dataset Full Stack: ingesta en streaming desde HuggingFace con filtrado en cascada
(lenguaje → dominio → calidad → deduplicación MinHash) hasta ~2.5–3 GB de código limpio.
