# Fase 01 — Dataset Full Stack

**Fecha:** 2026-09-25 / 2026-09-26
**Estado:** ✅ Completa (con desviaciones documentadas abajo)
**Commit:** ver `avance/README.md`

## Objetivo según el plan

Ingesta en streaming desde HuggingFace, filtrada al dominio Full Stack en cascada
(lenguaje → dominio → calidad → deduplicación MinHash), hasta ~2.5–3 GB de código limpio en
shards `.jsonl` comprimidos, sin cargar nunca el corpus completo en RAM.

## Qué se hizo

- `src/data/filters.py`: funciones puras. Lenguaje por extensión (`.js .jsx .ts .tsx .py`), rechazo por
  ruta (`node_modules`, `dist`, `build`, lockfiles, `*.min.js`…), señales de dominio con etiqueta de
  framework, y calidad (200 B–100 KB, ratio alfanumérico > 0.25, longitud media de línea < 150,
  autogenerados). Devuelve el motivo de rechazo para poder medir cada etapa.
- `src/data/dedup.py`: MinHash (128 permutaciones) + LSH (16 bandas × 8 filas) sobre 5-gramas de
  tokens, Jaccard 0.8. Implementación propia con numpy porque `datasketch` no está instalado.
- `src/data/ingest.py`: descarga directa del parquet (`requests`), lectura por row-groups (`pyarrow`),
  filtro + dedup, escritura de `raw/shard-NNNNN.jsonl.gz`. **Reanudable**: manifiesto y dedup se guardan
  tras cada shard, y al arrancar se comprueba que ambos coinciden.
- `src/data/stats.py`: distribución por lenguaje/framework, rechazos y muestreo para inspección.
- `tests/test_filters.py`, `tests/test_dedup.py`: 16 pruebas (casos frontera de tamaño, minificados,
  autogenerados, rutas, casi-duplicados, recarga tras guardar, formato antiguo).

Fuente: `codeparrot/github-code-clean` (abierto, sin token). The Stack está tras puerta de términos.

## Resultado

| Métrica | Valor |
|---|---|
| Shards fuente procesados | 131 de 880 |
| Archivos vistos | 16.627.175 |
| Archivos conservados | 643.239 (3,87 %) |
| Código conservado | 2,750 GB (media 4,3 KB/archivo) |
| Tamaño en disco (`.jsonl.gz`) | 1,9 GB |

Por lenguaje: js 57,2 % · py 23,5 % · ts 15,9 % · tsx 3,3 %.

Por señal de dominio: node 42,9 % · django 15,5 % · react 15,4 % · angular 7,0 % · ts-app 5,2 % ·
express 3,7 % · python-http 2,8 % · flask 2,4 % · pydantic 1,5 % · nestjs 1,4 % · python-async 1,4 % ·
vue 0,7 % · nextjs 0,1 % · fastapi 0,0 % (120 archivos).

Rechazos: lenguaje 85,6 % · fuera de dominio 12,5 % · ruta 1,1 % · duplicado 0,3 % ·
demasiado pequeño 0,2 % · autogenerado 0,1 % · demasiado grande 0,1 %.

## Desviaciones respecto al plan (importantes)

**1. Dominio ampliado.** Con las firmas del plan (React, Next, NestJS, Express, FastAPI, Flask,
`useState`, `async def`, `@app.`) el rendimiento fue 4,8 MB por shard: 2,75 GB habrían exigido ~570
shards (~200 GB de descarga, ~14 h). Con aprobación del Director se añadieron señales de Vue, Angular,
Django, pydantic/SQLAlchemy, librerías HTTP de Python, Node (`fs`, `process.env`, `module.exports`,
`router.get`…) y TypeScript de aplicación (`interface`/`type`, `fetch`). Rendimiento: ~20–26 MB por
shard; 131 shards. **Consecuencia honesta:** el corpus es más «JS/Python de aplicación web» que
«Full Stack moderno». Las señales amplias (`node`, `django`, `ts-app`) aportan ~63 % de los archivos;
**Next.js (0,1 %) y FastAPI (<0,1 %) están casi ausentes**, y tsx es solo 3,3 %. Es una limitación
real de la fuente (github-code-clean es de hace años) que afectará a lo que el modelo sepa hacer.

**2. Solo `.d.ts`, sin filtro específico.** Los archivos de declaraciones de tipos (`.d.ts`, p. ej. de
DefinitelyTyped) entran como `ts` (~1,5 % de una muestra de 1.000). No se filtraron; se anota por si
se decide excluirlos antes de la Fase 03.

**3. Reserva de RAM: el índice de dedup se reescribió.** La primera versión guardaba las bandas LSH en
dicts de Python: 2,28 GB de RAM con 410 k archivos (0,21 GB eran firmas reales) y re-pickle de 500 MB
tras cada shard. El sistema mató la ingesta por falta de memoria al llegar a 1,80 GB. Ahora las bandas
son arrays numpy ordenados y solo se guardan las firmas (`dedup.npz`):

| | Antes | Después |
|---|---|---|
| RAM fija con 410 k archivos | 2,28 GB | 0,37 GB |
| Pico procesando un shard | no medido | 1,05 GB |
| Guardado por shard | 500 MB pickle | 0,2 s, 212 MB |

Se retomó desde el último shard consistente sin repetir trabajo. No puedo demostrar que el índice fuera
la única causa de la falta de memoria (había ~6 GB libres en la máquina), pero fue el mayor consumidor
medido.

**4. `datasets` no se usa en la ingesta.** Ya no ejecuta scripts de carga; se descarga el parquet
directo. El `fallback` previsto en el plan es ahora el camino principal.

## Verificación ejecutada

| Comprobación | Resultado |
|---|---|
| `pytest tests/ -v` | 16 pruebas en verde |
| `python src/data/stats.py` | 2,750 GB, distribución arriba |
| Inspección manual de 20 muestras aleatorias | Código real de apps web; sin minificados ni autogenerados. Las cabeceras de licencia son frecuentes |
| Casi-duplicados residuales | 1.000 archivos aleatorios (499.500 pares), Jaccard exacto: **0 pares ≥ 0,8**, máximo 0,677 |
| Dedup vs. manifiesto | Coinciden (índice = `kept` del manifiesto) |
| RAM | Pico 1,05 GB por shard; sin nuevo aborto por memoria en la reanudación |

Límite de la comprobación de duplicados: es una muestra de 1.000 archivos, no exhaustiva, y MinHash
solo estima Jaccard; puede dejar pasar algún par por encima del umbral.

## Siguiente

Fase 02 — Tokenizer BPE orientado a sintaxis de código: entrenar 16k y 32k sobre este corpus y
elegir con datos medidos (bytes/token × velocidad). Con `node`/`django` dominando el corpus, conviene
revisar en esa fase cuántos tokens reales salen de 2,75 GB frente a los 500 M del presupuesto.
