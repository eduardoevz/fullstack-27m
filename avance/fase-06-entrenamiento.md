# Fase 06 — Entrenamiento

**Fecha de cierre:** 2026-10-05 (inicio: 2026-09-26)
**Estado:** ✅ Completa en lo esencial (15.260 pasos). Una parte del plan **no se cumplió**: no hay muestras generadas durante el entrenamiento (ver Desviaciones)
**Commit:** ver tabla de `avance/README.md`

## Objetivo según el plan

Entrenar el modelo en sesiones reanudables hasta el paso 15.260, o hasta que la pérdida de validación se estanque durante 2.000
pasos. Revisar la curva de pérdida (train y validación), el tok/s sostenido y guardar muestras generadas cada 1.000 pasos en
`samples/`.

## Qué se hizo

- Entrenamiento completo con `entrenar.bat` (sesiones reanudables desde el último checkpoint; ver `docs/guia-sesiones.md`).
- Config de la Fase 5: 32.768 tokens por paso (4 micro-batches), AdamW, LR con calentamiento de 500 pasos, pico 6e-4 y coseno
  hasta 6e-5, evaluación en validación cada 500 pasos.
- Terminó en el paso **15.260 de 15.260** (≈ 500,0 M tokens). No se activó el criterio de parada anticipada.
- Resultado guardado fuera del repo (`C:/llm-fullstack-data/checkpoints/`): `best.pt` (paso 15.000, val 1,4770) y el checkpoint
  final `ckpt-0015260.pt`.
- Se versiona la curva de validación en `avance/fase-06-curva-validacion.csv` (30 puntos), porque `logs/train.csv` está ignorado por git.

## Resultados

| Métrica | Valor |
|---|---|
| Pasos / tokens | 15.260 / ≈ 500 M |
| Pérdida de entrenamiento final | 1,499 (ruidosa por paso; ~1,44–1,50 en los últimos 50 pasos) |
| Pérdida de validación final | **1,4770** (paso 15.000, el mejor; la última evaluación fue la mejor) |
| Pérdida de validación inicial | 3,4447 (paso 500) |
| tok/s medio | **1.103** (1.058 en pasos 0–5.000; 1.135 en 5.000–10.000; 1.117 en 10.000–15.260) |
| Cómputo puro estimado | 15.260 × 32.768 / 1.103 ≈ **5,25 días** (proyección de la Fase 5: 5,1 sin derate) |
| RAM máxima | 1,93 GB |
| Grad-norm | máx. 6,63 en el paso 10 (arranque); tras el calentamiento, máx. 1,02 (paso 510); sin picos después |

Validación cada 500 pasos: 3,44 → 2,33 (1.000) → 1,93 (2.000) → 1,70 (5.000)... → 1,54 (10.000) → 1,50 (12.000) → 1,4918
(13.000) → 1,4832 (14.000) → 1,4770 (15.000). Serie completa en `avance/fase-06-curva-validacion.csv`.

### Lectura de la curva

- **Sin sobreajuste visible:** train (~1,50) y val (1,477) terminan prácticamente iguales.
- **Rendimientos decrecientes al final:** las últimas 4 evaluaciones mejoraron 0,0086, 0,0040, 0,0029 y 0,0033. Es la cola normal
  de un LR coseno que llega a su piso (6e-5), no un estancamiento: en los últimos 2.000 pasos la validación mejoró 0,015, así que
  el criterio de parada anticipada no aplicaba.
- No hay divergencias ni saltos tras reanudar sesiones.

## Desviaciones y honestidad

1. **No hay muestras en `samples/` (incumplimiento del plan).** El plan pedía generar texto cada 1.000 pasos para documentar la
   evolución del modelo. `train.py` no lo implementa y no se hizo a mano; la carpeta quedó vacía. Esas muestras intermedias ya
   **no se pueden recuperar** (solo quedan el checkpoint final, el mejor y la curva). Lo que sí se podrá hacer en la Fase 7 es
   generar con el modelo final.
2. **Caídas de velocidad:** 48 ventanas de 10 pasos bajaron de 800 tok/s (mínimo 369), agrupadas en los pasos 2.740–3.050,
   4.590–4.610, 4.870–4.940 y 12.860–13.150 (más 280 y 770 sueltos). No quedó registro de la causa (el log CSV no guarda los
   avisos térmicos de consola ni marcas de tiempo), así que **no puedo afirmar si fue calor, otra aplicación o Windows**. Se
   recuperó sola; no afectó a la pérdida.
3. **Duración en reloj no medida con exactitud:** el CSV no tiene marcas de tiempo. Por fechas de archivos fue del 26-sep al
   5-oct (≈ 9 días de calendario) en varias sesiones; las 5,25 días de cómputo son una estimación, no un cronómetro.
4. **Ruta de `Ctrl+C`/parada limpia:** `parar.bat` existe pero aquí no registré cuántas veces se usó ni si todas las paradas fueron limpias.
   La curva no muestra ninguna discontinuidad asociada.
5. **Calidad real del modelo aún sin medir.** Una pérdida de validación de 1,477 (≈ perplejidad 4,4 sobre el vocabulario de
   16k) es un buen indicador, pero no dice si el código generado es sintácticamente válido ni útil. Eso es la Fase 7.
6. Siguen abiertas las del corpus (Next.js y FastAPI casi ausentes, archivos autogenerados, `.d.ts`) y el nombre `27m`.

## Siguiente

Fase 07 — Inferencia, evaluación (validez sintáctica sobre 100 fragmentos y suite de 20 prompts), demo CLI y README final.
No arranca sin aprobación del Director.
