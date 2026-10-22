# Fase 14 — Nexora: interfaz web local

**Estado:** ✅ Completa. Interfaz web local del modelo v2 final (`hito-0018310.pt`), probada en Chrome con el modelo real.

## Objetivo

Dar al modelo una interfaz propia, como la de cualquier modelo de IA, para el portafolio: quien la vea debe notar que el autor construyó su primer modelo de IA.
Decisiones del Director: nombre **Nexora**, acceso **local** (modelo real en el equipo) con GIF y capturas reales, estilo **editor de código** oscuro, con completar, relleno en el cursor (FIM), ajustes y ficha del modelo.

**Desviación acordada en la conversación:** el Director pidió suavizar los límites del modelo en la ficha. No se hizo: la ficha pone primero los logros y luego las limitaciones con las cifras
medidas (tareas pass@1 2 %, parseo FIM 35,5 % frente a 44,5 %, suite de 20 prompts 55 % frente a 63 %), redactadas como criterio de ingeniería.

## Qué se construyó

| Archivo | Función |
|---|---|
| `src/web/server.py` | Servidor local (`http.server`, sin dependencias nuevas): `build_prompt` (FIM con `<CURSOR>`, normal con veto de marcadores), `parse_request` (validación), `Decoder` (UTF-8 sin partir), `POST /api/complete` por SSE, `GET /api/info`, estáticos protegidos contra rutas fuera de `static/`, un solo candado de generación (429 si está ocupado), aborta si el navegador se desconecta |
| `src/web/static/` | `index.html`, `app.css`, `app.js`: editor con resaltado propio (JS/TS/JSX/Python), tramo del modelo en otro color, botones Completar y Rellenar, Detener, ajustes (lenguaje, temperatura, tokens, penalización), estadísticas por generación, ejemplos, pestaña «Ficha del modelo» |
| `src/web/modelcard.json` | Cifras de la ficha, copiadas de `benchmarks/fim_eval_18310.json`, `v2_final_syntax512_rep1.1.json`, `v2_final_tasks_rep1.1.json` y `avance/fase-11-entrenamiento-fim.md` |
| `demo_nexora.bat` | Lanza el servidor con el modelo final y abre el navegador (`http://127.0.0.1:8765`) |
| `tests/test_web_server.py` | 26 tests: prompt FIM/normal, validación (15 entradas inválidas y límites), decodificador, servidor real con SSE, 400, rutas 404 y recorrido de directorios |
| `samples/nexora-demo.gif`, `nexora-editor.jpg`, `nexora-ficha.jpg` | Capturas reales |

## Verificación

- **Tests:** `python -m pytest tests -q` → **354 passed** (322 + 6 de la medición previa de la Fase 12 + 26 de Nexora).
- **Navegador (Chrome, modelo real):** Rellenar en el cursor (0,2–0,4 s, 18 a 31 tokens), Completar (200 tokens en 2,2 s, 91,6 tok/s; 77–81 tok/s en los rellenos), ficha con las cifras, Detener (la generación se corta y el servidor queda libre: una petición posterior por `curl` responde `done`), consola sin errores.
- **No verificado:** la vista en móvil (la ventana de Chrome no bajó de su ancho mínimo); el CSS tiene una regla para menos de 900 px, pero no se vio en pantalla.

## Cómo se ve la honestidad del modelo en la interfaz

- Los rellenos de la captura son reales y mediocres: el de `sumar` escribe `var b = a + b; return sum(b, b);` y una llave de más.
- Si un relleno agota los tokens sin cerrar, la barra de estado lo dice («No encontró dónde parar: es una limitación conocida del modelo»).

## Fuera de alcance

Alojamiento público, cuentas, historial, multiusuario, cuantización y reentrenamiento.
