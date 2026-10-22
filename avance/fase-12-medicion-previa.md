# Fase 12 — Medición previa (recocido de dominio y contexto 1.024)

**Estado:** 🔎 Solo medición previa. La Fase 12 **no se ejecutó** y sigue pendiente y opcional; requiere aprobación del Director.

Antes de gastar 1–2 días de cómputo se midió si hacía falta: ¿el contexto de 512 tokens limita al modelo?, ¿Next.js y FastAPI siguen flojos? Modelo medido: `hito-0018310.pt` (v2 final, 18.310 pasos).

## Herramienta

- `src/eval/context_probe.py` (+ `tests/test_context_probe.py`, 6 tests). Las reglas se fijaron en el docstring **antes** de medir.
- Mide la pérdida de los últimos 64 tokens de una ventana con contexto de 64/128/256/384/512 tokens (400 documentos de validación, hasta 1.500 ventanas), y la pérdida por dominio frente a la mezcla.
- Reglas: «limita» si la ganancia 256→512 ≥ 0,05 nats, o 384→512 ≥ 0,015 con ≥ 25 % de los tokens de validación más allá de la posición 512; «no limita» si ambas < 0,008. Dominio «flojo» si su pérdida supera 1,10× la de la mezcla.
- Resultados: `benchmarks/v2_context_probe.json` (principal) y `benchmarks/v2_context_probe_pos768.json` (control).

## Contexto de 512 tokens

| Contexto | Principal (tokens 449–512) | Control (tokens 705–768) |
|---|---|---|
| 64 | 2,0590 | 2,0330 |
| 128 | 1,4467 | 1,4935 |
| 256 | 1,2671 | 1,2671 |
| 384 | 1,1941 | 1,1945 |
| 512 | 1,2323 | 1,1496 |
| Ganancia 256→512 | 0,0347 | 0,1174 |
| Ganancia 384→512 | −0,0383 | 0,0449 |
| Veredicto por la regla | no limita | limita |

- El 58,4 % de los tokens de validación está en la posición 512 o más de su documento.
- **Las dos mediciones se contradicen.** En la principal, 512 sale peor que 384; sospecho que es un artefacto porque solo la ventana de 512 incluye el inicio del documento, pero no lo comprobé. El control, donde ninguna ventana toca el inicio, muestra que el contexto aún ayuda en el borde (−0,045 nats, ≈ 4 %).
- Lectura: hay indicio de que más contexto ayuda, de forma modesta. No se puede estimar la ganancia de pasar a 1.024; probablemente es menor que 0,045.

## Dominios (pérdida del v2 final)

| Dominio | v1 | v2 | Frente a la mezcla (1,3874) |
|---|---|---|---|
| FastAPI | 1,801 | 1,4555 | +5 % |
| NestJS | 1,400 | 1,1119 | −20 % |
| Next.js | 1,568 | 1,2412 | −11 % |
| React | 1,457 | 1,3163 | −5 % |

Ninguno pasa el umbral de +10 %: no hay dominio «flojo».

## Conclusión

- **Recocido de dominio:** sin motivo; ningún dominio queda flojo.
- **Contexto 1.024:** indicio modesto y no cuantificable; costaría ≈ +13 % de cómputo por token y 1–2 días.
- **Recomendación:** no hacer la Fase 12 ahora. Si se quisiera el contexto largo, sería un experimento corto y aparte.
