@echo off
rem Fase 11 (entrenamiento continuo v2, ~600 M tokens) desde la prueba FIM. Aborta solo si FIM no mejora (ver config). No toca checkpoints de v1.
rem Para parar sin perder nada: ejecuta parar.bat (o pulsa Ctrl+C aqui). Cierra la ventana solo tras "Checkpoint guardado".
cd /d "%~dp0"
title Entrenamiento LLM Full Stack - Fase 11 (v2 largo)
if exist "C:\llm-fullstack-data\checkpoints_v2_long\ckpt-*.pt" (
    echo Reanudando el entrenamiento v2 desde el ultimo checkpoint...
    python src\train.py --config config\model_v2_long.json --log logs\train_v2_long.csv --resume
) else (
    echo Empezando el entrenamiento v2 desde el checkpoint de la prueba FIM...
    python src\train.py --config config\model_v2_long.json --log logs\train_v2_long.csv
)
echo.
echo El entrenamiento se detuvo. Puedes cerrar esta ventana.
pause
