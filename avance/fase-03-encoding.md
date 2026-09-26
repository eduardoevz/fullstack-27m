# Fase 03 — Codificación a memmap binario

**Fecha:** 2026-09-26
**Estado:** ✅ Completa
**Commit:** ver `avance/README.md`

## Objetivo según el plan

Tokenizar el corpus una sola vez y guardarlo como `uint16` plano (`train.bin`, `val.bin`) para que el
entrenamiento lo lea con `np.memmap` con RAM constante, con documentos separados por `<|endoftext|>`
y barajados a nivel de documento.

## Qué se hizo

- `src/data/encode.py`, en dos etapas:
  - **A.** Tokeniza todo el corpus por lotes de 2.000 documentos a un archivo temporal `_all.bin` y guarda un índice por
    documento (offset, longitud, framework, lenguaje, hold-out). Cada documento se enmarca a nivel de ids:
    `<|file|> <|lang_x|> … <|endoftext|>` (jsx→js, tsx→ts).
  - **B.** Elige los tokens de entrenamiento con un tope común K por clase de framework (*water-filling*), sin duplicar
    archivos, baraja a nivel de documento y escribe `train.bin` y `val.bin` leyendo `_all.bin` por bloques. Después borra
    el temporal y escribe `tokens/meta.json` con la mezcla antes y después.
- `tests/test_encode.py`: 6 pruebas (enmarcado, reparto de K, selección determinista y sin duplicados, escritura por
  bloques, y un extremo a extremo con determinismo por hash y repos de val disjuntos de train). Total del repo: **39 en verde**.
- **Decisión del Director aplicada:** reequilibrar la mezcla sin duplicar.

## Resultados medidos

| Métrica | Valor |
|---|---|
| Documentos codificados | 643.239 (0 omitidos por contener tokens especiales) |
| **Tokens reales del corpus** | **822,6 M** (la estimación de la Fase 2 era ~826 M: −0,4 %) |
| `train.bin` | 499.993.466 tokens, 395.214 docs, 1,00 GB |
| `val.bin` | 4.434.965 tokens, 3.877 docs, 8,9 MB |
| Tiempo de codificación | ~10 min (~1,4 M tokens/s) |
| RAM al terminar | 0,38 GB (el pico intermedio no se midió) |

Los 822,6 M tokens superan los 500 M del presupuesto, así que **no hay repetición de época**: el entrenamiento verá 500 M
tokens distintos, elegidos entre 822,6 M.

### Mezcla antes y después (tokens de train; el «antes» excluye el hold-out)

| Clase | Antes | Después | Fracción conservada | % antes → % después |
|---|---|---|---|---|
| node | 343,1 M | 89,9 M | 0,26 | 42,1 → 18,0 |
| django | 152,0 M | 89,9 M | 0,59 | 18,7 → 18,0 |
| react | 79,7 M | 79,7 M | 1,00 | 9,8 → 15,9 |
| ts-app | 59,4 M | 59,4 M | 1,00 | 7,3 → 11,9 |
| python-http | 41,9 M | 41,9 M | 1,00 | 5,1 → 8,4 |
| angular | 34,0 M | 34,0 M | 1,00 | 4,2 → 6,8 |
| flask | 25,2 M | 25,2 M | 1,00 | 3,1 → 5,0 |
| express | 23,5 M | 23,5 M | 1,00 | 2,9 → 4,7 |
| pydantic | 21,1 M | 21,1 M | 1,00 | 2,6 → 4,2 |
| python-async | 20,2 M | 20,2 M | 1,00 | 2,5 → 4,0 |
| nestjs | 8,5 M | 8,5 M | 1,00 | 1,0 → 1,7 |
| vue | 5,9 M | 5,9 M | 1,00 | 0,7 → 1,2 |
| **nextjs** | **0,70 M** | **0,70 M** | 1,00 | **0,09 → 0,14** |
| **fastapi** | **0,10 M** | **0,10 M** | 1,00 | **0,01 → 0,02** |

Por lenguaje en `train.bin` (tokens): py 39,7 % · js 34,0 % · ts 22,2 % · tsx 4,1 %.

**Honestidad sobre el reequilibrio:** recortó `node` (42 → 18 %) y `django`, y dio más peso relativo a TypeScript, Angular,
Express, NestJS, Flask y pydantic. **No resolvió lo de Next.js y FastAPI**: ya se conservaban enteros y siguen siendo 0,14 % y
0,02 % del entrenamiento (0,7 M y 0,1 M tokens). Recortar lo dominante no crea datos que no existen. El modelo verá muy pocos
ejemplos de esos dos frameworks; para cambiarlo hace falta otra fuente de datos con más Next.js/FastAPI. Queda como decisión
abierta para el Director.

## Desviaciones respecto al plan

- **El plan decía «500 M de tokens al azar barajados»**; se sustituyó, por decisión del Director, por selección
  reequilibrada. Sigue sin duplicar ningún documento.
- **`val.bin` tiene 4,43 M tokens, no 5 M.** El hold-out (1 % de los repos) aplicado con las mismas fracciones de retención que
  train da 4,43 M; el objetivo de 5 M era un tope, no un mínimo garantizado.
- **La estimación previa de K (~95 M) no acertó**: salió 89,9 M, porque calculé las proporciones por número de documentos y no
  por tokens.
- **La etiqueta `framework` es aproximada:** es la primera coincidencia por prioridad de la Fase 01 (un archivo con React y
  Node cuenta como una sola clase).

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| `pytest tests/ -v` | 39 en verde |
| `max(token_id) < 16.384` (escaneo completo por bloques) | train 16.383, val 16.382 ✓ |
| Tamaño en bytes = 2 × tokens | ✓ en train y val |
| Nº de `<|endoftext|>` = nº de `<|file|>` = nº de docs | 395.214 = 395.214 = 395.214 (train); 3.877 (val) ✓ |
| Empieza con `<|file|>`, termina con `<|endoftext|>` | ✓ |
| `<|lang_x|>` tras cada `<|file|>` (primeros 20 M tokens de train) | ✓ |
| 3 fragmentos aleatorios decodificados | Código JS/TS legible (Node, webpack, tipos) |
| Repos de val disjuntos de train | Garantizado por construcción y comprobado con `assert` en el código y en el test extremo a extremo |
| Determinismo | Dos ejecuciones del test producen el mismo hash de `train.bin` y `val.bin` |
| Conteo real de tokens vs. estimación | 822,6 M vs ~826 M |

Límite de la verificación: la comprobación de `<|lang_x|>` tras `<|file|>` cubre solo los primeros 20 M tokens de `train.bin`,
no los 500 M; el recuento global de `<|file|>` = `<|endoftext|>` = docs sí es completo.

## Siguiente

Fase 04 — Arquitectura del Transformer (RMSNorm, RoPE, SwiGLU) con TDD: conteo de parámetros contra `param_count()` =
20,45 M, test de causalidad, pérdida inicial ≈ ln(16.384) ≈ 9,70 y sobreajuste de un batch. Ahí se valida por primera vez la
velocidad real (la cifra de 1.311 tok/s que sostuvo la elección de 16k viene de un prototipo).
