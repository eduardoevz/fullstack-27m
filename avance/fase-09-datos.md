# Fase 09 — Datos: más Next.js/FastAPI y menos ruido

**Fecha:** 2026-10-07 al 2026-10-08
**Estado:** ✅ Completa según el plan de la fase, con **tres reservas**: el piso de FastAPI (≥ 3 %) **no se alcanza** (2,4 %), el filtro por
perplejidad se **sustituyó** por reglas deterministas, y el Next.js obtenido es casi todo **Pages Router (2023)**, no App Router. La puerta
de cantidad (≥ 50 M tokens de Next.js + FastAPI) se cumple; la verificación manual fue una **inspección de 50 archivos**, no una lectura completa.
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

`docs/plan-mejoras-modelo.md` (Fase 9): sin reentrenar, producir un corpus v2 más limpio y mejor mezclado: limpieza determinista, reutilizar los
sobrantes de v1, una fuente moderna para Next.js/FastAPI/NestJS/TSX, un conjunto de validación por dominio, y la puerta «≥ 50 M tokens de
Next.js + FastAPI de calidad verificada a mano en una muestra de 50 archivos».

## Qué se hizo

- **Token de Hugging Face** en `.env` (excluido por `.gitignore`; comprobado con `git check-ignore` y con una búsqueda del token en el repo).
  `src/data/hf_auth.py` lo lee del entorno o del `.env`. Los términos de `bigcode/starcoderdata` los aceptó el Director en la web.
- **Limpieza determinista** (`src/data/filters.py`: `clean_reject`, `strip_license_header`, `strip_metadata_tags`, `clean_document`). Reglas: `.d.ts`,
  plantillas EJS/Handlebars, Python 2, Python que no parsea, blobs base64/hex, sourcemaps embebidos, mojibake, etiquetas de metadatos, cabeceras
  de licencia (≥ 4 líneas con palabras de licencia). `filter_file` no cambió, para no alterar la ingesta ya hecha.
- **Ingesta moderna** (`src/data/ingest_modern.py`): StarCoderData (directorios `typescript/` y `python/`, 86 shards), solo Next.js, FastAPI, NestJS
  y React en `.tsx`; topes de 400 MB (React) y 250 MB (NestJS); dedup contra el corpus ya ingerido (643.239 docs) y contra lo nuevo. Reanudable.
  Corrió 86/86 shards en ~14 h de reloj (4 min por shard con el equipo libre; 8–10 min cuando competía con otros procesos).
- **Índice de documentos de v1** (`src/data/doc_index.py`): reconstruye exactamente lo que vio v1 (499.993.466 tokens y 395.214 docs, **idéntico a
  `meta.json`**). Para ello la Etapa B de `encode.py` se extrajo a `choose_split` (misma función para v1 y para el índice).
- **Mezcla v2** (`src/data/build_v2.py`): une `raw/` y `raw_modern/`, re-limpia todo, reparte por (framework, tsx/otro) con **pisos** por dominio
  (`plan_mix`), prefiere documentos que v1 no vio, y escribe `train.bin`, `val.bin`, `val_<dominio>.bin` y `train_docs.npz` (offset/longitud de cada
  documento, para la Fase 10) en `tokens_v2/`.
- **Pérdida de v1 por dominio** (`src/eval/domain_loss.py`).
- **Filtro por perplejidad** (`src/eval/doc_ppl.py`): construido y calibrado, **no aplicado** (ver desviaciones).
- **Tests:** de 219 a **269 en verde** (cada grupo escrito antes del código).

## Resultados

### Cuánto ruido había (todo el corpus v1, 822,6 M tokens; `benchmarks/leftover_report.json`)

| Regla | Tokens | % |
|---|---|---|
| Python 2 | 28,7 M | 3,5 % |
| `.d.ts` | 17,8 M | 2,2 % |
| plantillas | 11,7 M | 1,4 % |
| Python que no parsea | 4,5 M | 0,5 % |
| otras | 0,4 M | 0,05 % |
| **Total** | **63,1 M** | **7,7 %** |

v1 se entrenó con ese 7,7 % dentro de `train.bin`. No cambia lo entrenado, pero conviene recordarlo al comparar v1 con v2.

### Sobrantes de v1 (no usados en sus 500 M)

315,4 M tokens (299,8 M tras limpiar), **solo `node` (253,3 M) y `django` (62,2 M)**. Todas las demás clases entraron enteras en v1, así que los sobrantes
**no aportan Next.js ni FastAPI**; sirven para que v2 no repita documentos de `node`/`django`.

### Fuente moderna (86 shards, tras limpieza y dedup)

| Framework | Documentos | Código | Notas |
|---|---|---|---|
| Next.js | 91.941 | 261 MB | casi todo Pages Router |
| FastAPI | 16.441 | 50,2 MB | todo el dataset de Python recorrido |
| NestJS | 127.062 | 250 MB | tope |
| React/TSX | 151.161 | 400 MB | tope |

### Mezcla v2 (`tokens_v2/`, `benchmarks/data_v2_mix.json`)

- Entrenamiento: **599.993.722 tokens** en 603.302 documentos (335.124 de v1, 268.178 modernos). Validación: 4,997 M tokens (5.415 docs).
- **20 %** de los tokens de entrenamiento son documentos que v1 nunca vio.

| Grupo | v1 | v2 | Piso | Estado |
|---|---|---|---|---|
| Next.js | 0,14 % | **11,4 %** (68,4 M tokens) | ≥ 5 % | ✅ |
| FastAPI | 0,02 % | **2,4 %** (14,5 M tokens) | ≥ 3 % | ❌ no hay más material |
| tsx | 4,1 % | **22,1 %** | ≥ 15 % | ✅ |

Por lenguaje: py 28,6 %, ts 25,1 %, js 24,3 %, tsx 22,1 %.

### Pérdida de v1 (paso 15.000) por conjunto de validación (`benchmarks/v1_domain_loss.json`; ventanas de 512, ≤ 1.500)

| Conjunto | Pérdida | Ventanas |
|---|---|---|
| FastAPI | **1,8006** | 275 |
| Next.js | 1,5682 | 1.275 |
| Validación mezcla v2 | 1,5128 | 1.500 |
| Validación mezcla v1 | 1,5125 | 1.500 |
| React | 1,4566 | 1.500 |
| NestJS | 1,4001 | 1.500 |

FastAPI y Next.js son los dominios más difíciles para v1, que casi no los vio. La validación general de v2 y la de v1 casi coinciden, así que la mezcla
nueva no es más fácil ni más difícil en promedio. Estas cifras son comparables entre sí, **no** con el 1,477 de v1 (otro muestreo). FastAPI, con 275
ventanas, es la medida más ruidosa.

## Lectura de la puerta (criterio del plan)

> «≥ 50 M tokens de Next.js + FastAPI de calidad verificada a mano en una muestra de 50 archivos.»

- **Cantidad: se cumple.** El pool tiene 77,2 M (Next.js) + 14,5 M (FastAPI) = 91,7 M de tokens; el entrenamiento v2 incluye 68,4 + 14,5 = **82,9 M**.
- **Verificación manual: hecha de forma limitada.** Se sacaron 50 archivos al azar de `tokens_v2` (25 Next.js, 25 FastAPI), ya limpios. Se inspeccionaron las
  primeras líneas y marcadores (App/Pages Router, Pydantic v1/v2, hooks, tipos, etiquetas) de cada uno; **no se leyó cada archivo completo**.
  - Ninguno tenía etiquetas de metadatos ni era un esqueleto vacío; todos eran código real del framework (componentes, páginas, routers, tests).
  - **Modernidad: baja.** 0 de 25 de Next.js usan App Router (`use client`, `next/navigation`); casi todos usan `next/router`, `next/head`, `getStaticProps`.
    En FastAPI, 0 de 25 muestran marcadores de Pydantic v2 y varios usan el estilo v1 (`.dict()`, `orm_mode`, `@validator`).
  - Algunos archivos son marginales (p. ej. un servidor de detección de objetos que solo usa FastAPI como envoltorio; un fork de FastAPI con sus propios tests).
  - **Licencias: no verificables por archivo** (el dataset no trae licencia por documento). Se confía en que StarCoderData ya filtra por licencia permisiva;
    no lo comprobé.
- **Conclusión:** la mejora en *cantidad* de Next.js/FastAPI es real y grande (de 0,8 M a 83 M de tokens), pero el contenido corresponde al ecosistema de 2020–2023.
  Lo que ya no se puede esperar de esta fuente es App Router ni Pydantic v2.

## Desviaciones y honestidad

1. **Filtro por perplejidad sustituido por reglas deterministas** (decisión del Director tras ver los datos). Puntuar una muestra de 7.920 documentos con v1
   tardó 815 s (≈ 2.300 tokens/s); puntuar ~500k candidatos serían 6–14 h. Además, de los 14 peores de una lista de 40 que revisé a mano, ~4 eran ruido
   (un certificado, un sourcemap, dos con caracteres corruptos) y el resto código legítimo que a v1 le resulta raro (un `_app.tsx`, módulos con docstrings, comentarios
   en otro idioma). Es una lectura de 14 casos, no una medición de precisión. Las reglas nuevas (`blob`, `sourcemap`, `mojibake`) suman ~0,26 % de los caracteres.
2. **Fuente: StarCoderData, no The Stack v2.** The Stack v2 entrega referencias a Software Heritage (descarga más compleja). StarCoderData trae el contenido en parquet.
3. **Hallazgo grave durante la revisión manual:** el 48 % de los documentos modernos (47.963 de una muestra de 100.628) empezaban con etiquetas
   `<reponame>…<filename>…<gh_stars>…`. Se corrigió con `strip_metadata_tags` y la mezcla se **construyó tres veces**: las dos primeras corridas tenían ese ruido (la primera,
   además, un defecto en `plan_mix`, ver 4) y se descartaron. En la corrida final, 0 de 8.000 documentos decodificados al azar de `train.bin` contienen etiquetas
   (es una muestra, no una revisión exhaustiva), y 346 documentos con etiquetas en medio del archivo se rechazaron.
4. **Defecto de `plan_mix` corregido:** un piso inalcanzable (FastAPI) reforzaba a todos los pisos y metía todo el TSX de React (tsx = 35,7 %). Ahora un piso inalcanzable
   deja ese grupo entero sin reforzar a los demás (tsx = 22,1 %). Hay un test que lo reproduce.
5. **Piso de FastAPI sin cumplir (2,4 % frente a 3 %):** no hay más material en StarCoderData; se incluyó todo lo disponible. Cumplirlo exigiría otra fuente.
6. **Topes de React (400 MB) y NestJS (250 MB) fijados por mí** antes de ver los datos, para no inundar la mezcla; con ellos el tsx cumple su piso sin problema.
7. **Next.js en `.tsx` queda recortado a 65,5 M tokens de 74,4 M disponibles** por el tope común de la mezcla (~10 M de Next.js no entran).
8. **Las reglas se endurecieron durante la fase** (blob, sourcemap, mojibake, etiquetas) después de haber ingerido parte de los shards; por eso `build_v2` vuelve a
   aplicar `clean_document` a todo antes de codificar.
9. **Un dato mal estimado al inicio:** proyecté ~45 M de tokens para FastAPI; el pool real tiene 14,5 M de tokens (50 MB de código) tras limpieza, dedup y holdout.
10. **Seguridad:** el token de Hugging Face se pegó en el chat. Recomendación: **revocarlo y crear otro** en https://huggingface.co/settings/tokens. No está en ningún archivo versionado.
11. **No se midió** el efecto de v2 sobre ningún modelo: esta fase solo prepara datos. Las cifras de pérdida son la línea base de v1 por dominio.

## Siguiente

Fase 10 (formato: FIM y cierre de archivo) y Fase 11 (entrenamiento continuo con `tokens_v2/`). `train_docs.npz` ya trae los offsets de documento que necesitará la Fase 10.
Antes de la Fase 11, el plan pide una métrica funcional más sensible (acierto de la línea siguiente sobre código real de validación). Cada fase requiere aprobación del Director.
Si se quiere Next.js moderno (App Router) o FastAPI más abundante, hace falta otra fuente; The Stack v2 sería el candidato.
