# Plan: LLM Full Stack pre-entrenado desde cero en CPU

## Contexto

Construir, **desde cero y en CPU local**, un modelo de lenguaje causal especializado en autocompletado y generación de código Full Stack (React/Next.js/TypeScript en frontend; Node/Express/NestJS y Python/FastAPI/Flask en backend). El objetivo es una pieza de portafolio de **Ingeniería de ML**: lo que se demuestra no es el tamaño del modelo, sino el dominio del pipeline completo — datos, tokenizer, arquitectura, bucle de entrenamiento y optimización bajo restricciones de hardware reales.

### Hardware y entorno (medidos, no supuestos)

| Elemento | Valor medido |
|---|---|
| CPU | Intel i5-1135G7 — 4 núcleos / 8 hilos, 2.4 GHz base |
| RAM | 15.74 GB total (~6 GB libres en el momento de la medición) |
| Disco | 716 GB libres en C: |
| Python | 3.14.7 (`C:\Python314`) |
| PyTorch | **2.14.0+cpu ya instalado** |
| Vectorización | AVX-512 activo, MKLDNN disponible, `torch.get_num_threads() = 4` |
| Ya instalado | `numpy 2.5.2`, `requests 2.34.2` |
| Falta instalar | `tokenizers`, `datasets`, `tqdm` |
| Compilador C++ | **Ausente** (no hay `cl.exe`; VS Installer presente pero sin C++ Build Tools) |

### Mediciones de rendimiento que definen el plan

```
GEMM fp32  (1024³)                      216 GFLOPS
GEMM bf16  (1024³)                       72 GFLOPS   <-- 3x MAS LENTO

Config 48M (d=512, L=10, vocab 32k), B=8    13.51 s/paso ->   303 tok/s
Config 27M (d=384, L=8,  vocab 32k), B=16    7.75 s/paso ->  1057 tok/s
Config 21M (d=384, L=8,  vocab 16k), B=16    6.25 s/paso ->  1311 tok/s
```

**Dos hallazgos que cambian el diseño original:**

1. **La precisión mixta queda descartada.** Tiger Lake tiene AVX-512 pero **no** AVX512-BF16 ni AMX, así que bf16 se emula en software y resulta 3× más lento que fp32. Entrenaremos en fp32 puro. Este resultado negativo se documenta en el README: es exactamente el tipo de decisión guiada por medición que se espera de un ingeniero de ML.

2. **50M parámetros era el objetivo equivocado.** El óptimo Chinchilla para 50M son ~1.000M tokens ≈ **45 días** de esta laptop. A 27M parámetros, el presupuesto de 8 días rinde ~500M tokens = **19 tokens/parámetro**, prácticamente el óptimo Chinchilla (534M). Mismo tiempo de cómputo, modelo sustancialmente mejor.

### Decisiones tomadas por el Director del Proyecto

- **Escala:** ~27M parámetros bien entrenados en lugar de 50M infraentrenados.
- **Datos:** ingesta en streaming desde HuggingFace, filtrada al dominio Full Stack.
- **Presupuesto:** ~8 días de entrenamiento, particionable en sesiones con reanudación.

### Presupuesto objetivo

| Métrica | Valor |
|---|---|
| Parámetros | ~26.5 M (14.2 M sin embeddings) |
| Tokens de entrenamiento | **500 M** |
| Ratio | 19 tokens/parámetro |
| Throughput | 1057 tok/s → 63.9 M tok/día (con derate térmico 0.7) |
| Duración | **~7.8 días** de cómputo efectivo |
| Pasos de optimizador | ~15.260 (batch efectivo 32.768 tokens) |

---

## Decisiones de ingeniería

### Almacenamiento fuera de OneDrive — obligatorio

El proyecto vive en `C:\Users\eduem\OneDrive\Desktop\LLM-Eduardo`, es decir, **dentro de una carpeta sincronizada**. Un dataset de ~3 GB y checkpoints de ~430 MB reescritos cada 30 minutos provocarían sincronización continua, consumo de disco y —el riesgo real— **bloqueos de archivo de OneDrive durante la escritura de un checkpoint**, que abortarían un entrenamiento de días.

- El **código** permanece en la carpeta del proyecto (es pequeño y se versiona con git).
- Los **datos y checkpoints** van a `C:\llm-fullstack-data\`, fuera de OneDrive.
- Se crean junctions (`mklink /J`) para que las rutas relativas del repo sigan funcionando.

### Arquitectura: estilo Llama, no GPT-2

A igualdad de parámetros, RMSNorm + RoPE + SwiGLU entrena mejor y converge más rápido que LayerNorm + embeddings posicionales aprendidos, con coste de cómputo neutro. Además RoPE elimina el límite duro de contexto, lo que permite extrapolar más allá de 512 tokens en inferencia.

```
d_model      384          n_layers   8        n_heads   6  (head_dim 64)
ctx          512          vocab      32.768   FFN       1024 (SwiGLU)
Norm         RMSNorm      Pos        RoPE     Bias      ninguno
Embeddings   tied (entrada y salida comparten pesos)
```

### El cuello de botella es la cabeza de salida

Con `d=384` y vocab 32k, la proyección final (384 × 32.768) consume **~46% del cómputo total por token**. Por eso la elección de vocabulario no es cosmética: vocab 16k corre 24% más rápido pero comprime peor el texto. La decisión se toma **con datos medidos** en la Fase 2, no por intuición.

### Sin precisión mixta, sin `torch.compile` (por ahora)

- fp32 puro, por la medición de bf16 arriba.
- `torch.compile` con backend inductor requiere `cl.exe` en Windows, que no está instalado. Queda como mejora opcional con puerta de decisión explícita en la Fase 4.
- Las palancas de optimización reales en este hardware son: número de hilos, afinidad (`KMP_AFFINITY`), tensores contiguos, SDPA fusionado y tamaño de micro-batch.

---

## Fases

### Fase 0 — Cimientos y banco de pruebas

Estructura del repositorio, dependencias y —lo más importante— el **arnés de benchmark** que servirá de línea base para todas las decisiones posteriores.

**Archivos a crear:**
- `requirements.txt` — `tokenizers`, `datasets`, `tqdm` (torch y numpy ya están)
- `config/model_27m.json` — única fuente de verdad de los hiperparámetros
- `src/bench.py` — mide GFLOPS, tok/s y RAM pico para una config dada; escribe `benchmarks/baseline.json`
- `.gitignore`, `README.md` (esqueleto)
- Junctions: `data/` y `checkpoints/` → `C:\llm-fullstack-data\`

**Verificación:** `python src/bench.py --config config/model_27m.json` reproduce ~1057 tok/s (±10%) y guarda la línea base. `git init` + primer commit.

---

### Fase 1 — Dataset Full Stack

Ingesta en streaming desde HuggingFace, filtrada agresivamente al dominio. Nunca se carga el corpus completo en RAM: se procesa por lotes y se escribe a shards `.jsonl` comprimidos en disco.

**Filtros en cascada:**
1. **Lenguaje:** `.js`, `.jsx`, `.ts`, `.tsx`, `.py`
2. **Dominio** (por firmas en el contenido): `import React`, `next/`, `@nestjs`, `express()`, `from fastapi`, `from flask`, `useState`, `async def`, `@app.`
3. **Calidad:** 200 B – 100 KB; ratio alfanumérico > 0.25; longitud media de línea < 150; descartar minificados, lockfiles, `node_modules`, archivos autogenerados y build artifacts
4. **Deduplicación:** MinHash + LSH sobre 5-gramas (umbral Jaccard 0.8) — el código duplicado en GitHub es masivo y desperdicia cómputo directamente

**Archivos:** `src/data/ingest.py`, `src/data/filters.py`, `src/data/dedup.py`

**Objetivo:** ~2.5–3 GB de código limpio (suficiente para 500M tokens sin repetir época).

**Verificación:** `pytest tests/test_filters.py` con casos frontera; `src/data/stats.py` imprime distribución por lenguaje y framework; inspección manual de 20 muestras aleatorias.

**Riesgo:** `datasets` podría no tener wheel para Python 3.14. *Fallback:* descarga directa de los ficheros parquet con `requests` + `pyarrow`, que ya cubre el caso de streaming.

---

### Fase 2 — Tokenizer BPE orientado a sintaxis de código

BPE byte-level entrenado desde cero sobre **nuestro** corpus (no reutilizado), con pre-tokenizer adaptado a código: preserva indentación (tokens dedicados para 2/4/8 espacios), no rompe `camelCase` ni `snake_case` arbitrariamente, y trata operadores compuestos (`=>`, `===`, `?.`, `::`) como unidades.

Tokens especiales: `<|endoftext|>`, `<|file|>`, `<|lang_js|>`, `<|lang_ts|>`, `<|lang_py|>`, `<|pad|>`.

**Decisión medida — 16k vs 32k:** se entrenan **ambos** y se comparan en bytes/token sobre un conjunto de validación. Se elige el que maximice `bytes_por_token × velocidad_relativa`:

```
vocab 32k: 1057 tok/s  x  bytes/token medido
vocab 16k: 1311 tok/s  x  bytes/token medido   <- gana si comprime >= 81% que 32k
```

**Archivos:** `src/tokenizer/train_tokenizer.py`, `src/tokenizer/compare_vocabs.py`

**Verificación:** round-trip exacto (`decode(encode(x)) == x`) sobre 1000 archivos; reporte de compresión; confirmar que la indentación sobrevive al ciclo.

---

### Fase 3 — Codificación a memmap binario

El corpus se tokeniza una sola vez y se persiste como `uint16` plano. El entrenamiento lo lee con `np.memmap`: **el SO pagina bajo demanda y la RAM usada es constante independientemente del tamaño del dataset**. Es la solución real a la restricción de 16 GB.

- `data/train.bin` — 500M tokens × 2 B = **1.0 GB**
- `data/val.bin` — 5M tokens = 10 MB
- Documentos separados por `<|endoftext|>` y barajados a nivel de documento antes de concatenar

**Archivos:** `src/data/encode.py`

**Verificación:** el conteo de tokens coincide con lo esperado; `max(token_id) < vocab_size`; decodificar un fragmento aleatorio produce código legible.

---

### Fase 4 — Arquitectura del Transformer

**Archivos:** `src/model/gpt.py`, `src/model/rope.py`

Implementación limpia y autocontenida: `RMSNorm`, `RotaryEmbedding`, `CausalSelfAttention` (sobre `F.scaled_dot_product_attention`), `SwiGLU`, `Block`, `GPT`. Inicialización escalada por profundidad (`0.02/sqrt(2*n_layers)` en las proyecciones residuales).

**Verificación (TDD — los tests se escriben antes):**
- El conteo de parámetros coincide con la fórmula analítica y cae en 26.5M ± 0.5M
- **Test de causalidad:** alterar el token en la posición *t* no cambia ninguna salida en posiciones `< t`
- Forma de salida correcta; forward determinista con semilla fija
- Pérdida inicial ≈ `ln(vocab_size)` ≈ 10.4 — si no, hay un bug de inicialización
- Sobreajuste deliberado de un batch de 32 secuencias hasta pérdida < 0.1 (prueba de que los gradientes fluyen)

---

### Fase 5 — Bucle de entrenamiento optimizado para CPU

**Archivo:** `src/train.py`

| Hiperparámetro | Valor | Razón |
|---|---|---|
| Micro-batch | 16 × 512 = 8.192 tokens | máximo que cabe cómodo en RAM |
| Acumulación de gradiente | 4 | batch efectivo 32.768 tokens sin coste de memoria |
| Pasos totales | ~15.260 | 500M tokens / 32.768 |
| Optimizador | AdamW (0.9, 0.95), wd 0.1 | wd solo en matrices, no en normas/embeddings |
| LR | 6e-4 pico, coseno → 6e-5 | warmup lineal de 500 pasos |
| Recorte de gradiente | 1.0 | |
| Precisión | fp32 | bf16 medido 3× más lento |
| Hilos | 4 (= núcleos físicos) | 8 hilos con SMT degrada el rendimiento en GEMM |

**Puntos críticos de ingeniería:**
- **Checkpointing y reanudación reales.** Se guardan modelo, estado del optimizador, paso, RNG y posición en los datos cada 500 pasos. Un entrenamiento de 8 días *va* a interrumpirse; reanudar debe ser exacto, no aproximado.
- **Rotación de checkpoints:** conservar los 2 últimos + el mejor por pérdida de validación (~430 MB cada uno).
- **Carga de datos sin copia:** muestreo de offsets aleatorios sobre el memmap; sin `DataLoader` ni workers (en Windows el overhead de spawn no compensa para lecturas memmap).
- **Registro a CSV** (paso, pérdida, LR, tok/s, RAM, grad-norm) para graficar las curvas en el README.
- **Guardia térmica:** registrar tok/s por paso; una caída sostenida > 30% indica throttling y queda documentada.

**Puerta de decisión sobre `torch.compile`:** intentar `torch.compile(model)`. Si falla por ausencia de `cl.exe`, presentar la opción de instalar VS Build Tools (~3 GB) y medir la ganancia real antes de comprometer los 8 días. Sin ganancia medida, no se instala nada.

**Verificación:** entrenar 200 pasos; confirmar que la pérdida baja de ~10.4 a < 7; matar el proceso, reanudar y comprobar que la pérdida continúa sin salto; verificar que la RAM se mantiene plana.

---

### Fase 6 — Entrenamiento (~8 días)

Ejecución en sesiones, con reanudación entre ellas. Revisión periódica de:
- Curva de pérdida de entrenamiento y validación (la divergencia entre ambas indica sobreajuste → más datos o más regularización)
- Muestras generadas cada 1.000 pasos, guardadas en `samples/` para documentar la evolución del modelo — material excelente para el portafolio
- tok/s sostenido frente a la línea base de la Fase 0

**Criterio de parada:** 15.260 pasos, o pérdida de validación estancada durante 2.000 pasos.

---

### Fase 7 — Inferencia, evaluación y demo

**Archivos:** `src/sample.py`, `src/eval/syntax_check.py`, `src/eval/prompts.py`, `src/cli.py`

- **Generación:** muestreo con temperatura, top-k y top-p; caché KV para que el autocompletado sea interactivo
- **Métrica objetiva de validez sintáctica** — la métrica estrella del portafolio: generar 100 fragmentos y medir qué porcentaje parsea (`ast.parse` para Python, `node --check` para JS/TS). Es un número defendible, no una impresión subjetiva.
- **Suite de 20 prompts del dominio:** componente React con hooks, endpoint FastAPI con Pydantic, controlador NestJS, middleware de Express, ruta de Next.js App Router, etc.
- **Demo CLI de autocompletado:** el usuario escribe código y el modelo completa — la pieza demostrable del proyecto.
- **README final:** arquitectura, decisiones justificadas por medición (bf16, elección de escala, vocab), curvas de entrenamiento, resultados y limitaciones honestas.

**Verificación:** ejecutar la suite completa y registrar el porcentaje de validez sintáctica; grabar la demo CLI en funcionamiento.

---

## Verificación integral

Al terminar, esta secuencia debe funcionar de principio a fin:

```powershell
python src/bench.py --config config/model_27m.json     # linea base reproducible
pytest tests/ -v                                        # filtros, tokenizer, modelo
python src/data/stats.py                                # composicion del dataset
python src/train.py --config config/model_27m.json --max-steps 50 --dry-run
python src/sample.py --ckpt checkpoints/best.pt --prompt "export default function Nav("
python src/eval/syntax_check.py --ckpt checkpoints/best.pt --n 100
python src/cli.py                                       # demo interactiva
```

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| `datasets` sin wheel para Python 3.14 | Fallback a descarga directa de parquet con `requests` + `pyarrow` |
| OneDrive bloquea un checkpoint a mitad de escritura | Datos y checkpoints fuera de OneDrive (decisión de la Fase 0) |
| Throttling térmico de la laptop | Medido y registrado por paso; derate 0.7 ya incluido en el presupuesto |
| Interrupción del entrenamiento | Reanudación exacta, probada explícitamente en la Fase 5 |
| Dataset insuficiente tras el filtrado | Los filtros son parámetros de configuración; relajar el filtro de dominio amplía el corpus |
| Modelo infraentrenado pese a todo | 19 tokens/param es casi el óptimo Chinchilla; expectativa honesta documentada en el README |

## Lo que este proyecto NO es

Un modelo de 27M parámetros con 500M tokens produce código **sintácticamente correcto y estructuralmente reconocible**, no código listo para producción. El README lo dirá con claridad. El valor de portafolio está en el pipeline de ingeniería y en las decisiones medidas — no en fingir que compite con Copilot.
