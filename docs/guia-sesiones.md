# Guía de sesiones de entrenamiento (Fase 6)

| Quiero… | Hago… |
|---|---|
| Empezar o continuar entrenando | Doble clic en `entrenar.bat` (reanuda solo desde el último checkpoint) |
| Parar sin perder nada | Doble clic en `parar.bat`; esperar a que la ventana diga «Checkpoint guardado» (hasta ~30 s) |
| Ver cómo va | Doble clic en `estado.bat` |
| Apagar la laptop | Primero `parar.bat`, esperar el mensaje, luego apagar |

- Checkpoint automático cada 20 pasos (~10 min): un corte de luz pierde como mucho ese tiempo.
- Los checkpoints están en `C:/llm-fullstack-data/checkpoints/` (2 últimos + `best.pt`); el registro, en `logs/train.csv`.
- La ventana de `entrenar.bat` es independiente de Claude Code: puede cerrarse la conversación y sigue entrenando.
- Antes de cada sesión: cargador conectado, suspensión desactivada, Windows Update pausado, apps pesadas cerradas y la laptop ventilada.
- Si el registro muestra `[AVISO termico]`, la velocidad cayó más de 30 %: probable calentamiento.
