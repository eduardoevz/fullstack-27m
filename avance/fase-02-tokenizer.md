# Fase 02 — Tokenizer BPE orientado a sintaxis de código

**Fecha:** 2026-09-26
**Estado:** ✅ Completa (con desviaciones y límites documentados abajo)
**Commit:** `d0017a6`

## Objetivo según el plan

BPE byte-level entrenado desde cero sobre el corpus propio, con pre-tokenizer adaptado a código
(sangría preservada, identificadores enteros, operadores compuestos como unidad), tokens especiales,
y comparación medida 16k vs 32k para elegir vocabulario.

## Qué se hizo

- `src/tokenizer/pretokenizer.py`: `Split(regex)` + `ByteLevel(use_regex=False)`. El regex mantiene como unidad
  (1) la sangría al inicio de línea (`\n` + espacios/tabs), (2) operadores compuestos (`===`, `!==`, `=>`, `?.`, `??`,
  `::`, `**`, `...`…), (3) identificadores completos, (4) números; un espacio previo se pega a la palabra siguiente.
  Tokens especiales con ids fijos 0–5: `<|endoftext|>`, `<|file|>`, `<|lang_js|>`, `<|lang_ts|>`, `<|lang_py|>`, `<|pad|>`.
- `src/tokenizer/corpus.py`: lectura en streaming, **hold-out por repo** (crc32, ~1 %) excluido del entrenamiento del
  tokenizer, y muestreo aleatorio uniforme por documento (misma mezcla de lenguajes que el corpus). La misma partición
  debe reutilizarse para `val.bin` en la Fase 03.
- `src/tokenizer/train_tokenizer.py`: entrena con una muestra de 700 MB. 145 s por vocabulario; RAM baja (≤ 0,7 GB
  observados en muestreo puntual, no un pico medido con precisión).
- `src/tokenizer/compare_vocabs.py`: métricas sobre el hold-out y regla de decisión del plan. Resultados en
  `benchmarks/tokenizer_comparison.json` (versionado).
- `tests/test_tokenizer.py`: 17 pruebas nuevas (round-trip exacto con tab, CRLF, unicode y emoji; sangrías, operadores e
  identificadores; ids especiales; hold-out; coherencia config↔tokenizer). Total del repo: **33 en verde**.

## Resultados (hold-out: 5.000 documentos, 20,1 MB, de repos no vistos)

| | 16.384 | 32.768 |
|---|---|---|
| bytes/token | 3,338 | 3,451 |
| tokens/documento | 1.203 | 1.164 |
| js / py / ts / tsx (bytes/token) | 3,20 / 3,54 / 3,36 / 3,40 | 3,29 / 3,69 / 3,46 / 3,51 |
| Round-trip exacto (1.000 archivos) | 0 fallos | 0 fallos |
| Sangría de 2/4/8/12 espacios, tab, 2 tabs | 1 token cada una | 1 token cada una |
| Tokens estimados del corpus completo | ~826 M | ~799 M |
| Velocidad del modelo (Fase 0, prototipo) | 1.311 tok/s | 1.057 tok/s |
| **Puntuación = bytes/token × tok/s** | **4.376** | 3.647 |

El 16k comprime el **96,7 %** de lo que comprime el 32k (umbral del plan: 81 %) → **gana 16.384**.

## Decisión y consecuencias

Se adopta vocab 16.384 (`config/model_27m.json` y ruta `tokenizer-16384.json`).

- **El modelo baja de 26,75 M a 20,45 M parámetros** (14,2 M sin embeddings, igual que antes). El nombre `27m` del repo y del
  config ya no es exacto; se conserva por estabilidad de rutas y se explica en el README.
- **Pregunta abierta de la Fase 01, resuelta:** 2,75 GB dan ~826 M tokens, **más** que los 500 M del presupuesto. No hay
  que repetir época ni reducir tokens; sobran ~326 M. Con 20,45 M parámetros son 24,4 tokens/parámetro (antes 19).
- Estimación de duración (no medida): 500 M / (1.311 tok/s × 0,7 de derate) ≈ 6,3 días, frente a los 7,8 anteriores.

## Límites y advertencias honestas

1. **El 1.311 tok/s viene de un prototipo desechable**, no del modelo del repo. La decisión 16k depende de esa cifra; se
   valida en la Fase 04. Aunque la velocidad fuera igual, el 16k solo perdería un 3,3 % de compresión.
2. **La estimación de ~826 M tokens es una extrapolación**: bytes/token medido en el 1 % de repos, aplicado a todo el corpus
   más 3 tokens especiales por documento. Con hold-out por repos pequeño puede desviarse unos puntos. Se confirma en la
   Fase 03, que codifica el corpus entero.
3. **Solo se evaluaron 5.000 de los ~6.400 documentos del hold-out** y el round-trip en 1.000 (lo que pedía el plan).
4. **No se listaron los merges por lenguaje** que pedía el plan; sí se inspeccionaron los tokens más largos.
5. **Desperdicio menor de vocabulario:** 163 tokens (1,0 %) del 16k son solo blancos; 62 miden más de 16 caracteres
   (tramos de sangría de hasta 73). No justifica cambiar el diseño.
6. **Se coló código generado:** hay tokens como `__WEBPACK_IMPORTED_MODULE_` y `python_2_unicode_compatible`. El filtro de
   autogenerados de la Fase 01 solo mira la cabecera; el bundle de webpack pasa. Afecta a una fracción pequeña del
   vocabulario y del corpus, pero ocupa huecos de vocab y entrenamiento del modelo.
7. **Agrupación de símbolos no evaluada:** cada símbolo suelto es su propio pre-token (`);` son 2 tokens que BPE no puede
   fundir). Es coherente con «operadores como unidad», pero una variante que agrupe tramos de símbolos podría comprimir
   más; no se midió.

## Desviaciones respecto al plan

- El entrenamiento del tokenizer **no antepone** `<|file|>` ni `<|lang_x|>` a los documentos: se añaden al codificar en la
  Fase 03. Se decidió así para no depender de cómo trata el trainer los tokens especiales presentes en el texto.
- El README se actualizó (parámetros y vocabulario) y el config cambió de vocab 32.768 a 16.384: consecuencia directa de
  la puerta de decisión que el propio plan definió.

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| `pytest tests/ -v` | 33 en verde |
| Round-trip `decode(encode(x)) == x`, 1.000 archivos del hold-out, ambos vocabularios | 0 fallos |
| La sangría sobrevive al ciclo | 2/4/8/12 espacios y tab: 1 token cada una |
| `Tokenizer.get_vocab_size()` == `model.vocab_size` del config | Coincide (16.384) |
| Conteo analítico de parámetros con vocab 16.384 | 20,454 M |

## Siguiente

Fase 03 — Codificación a memmap binario: tokenizar el corpus una sola vez con `<|file|><|lang_x|>…<|endoftext|>`,
`train.bin` (uint16) y `val.bin` con el mismo hold-out por repo. Ahí se mide el conteo real de tokens.
