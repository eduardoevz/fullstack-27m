@echo off
rem Prueba corta de la Fase 10 (FIM, 1.000 pasos) desde los pesos de v1. No toca checkpoints de v1.
rem Para parar sin perder nada: ejecuta parar.bat (o pulsa Ctrl+C aqui). Cierra la ventana solo tras "Checkpoint guardado".
cd /d "%~dp0"
title Entrenamiento LLM Full Stack - Fase 10 (FIM)
if exist "C:\llm-fullstack-data\checkpoints_v2\ckpt-*.pt" (
    echo Reanudando la prueba FIM desde el ultimo checkpoint...
    python src\train.py --config config\model_v2.json --log logs\train_v2.csv --resume
) else (
    echo Empezando la prueba FIM desde los pesos de v1...
    python src\train.py --config config\model_v2.json --log logs\train_v2.csv
)
echo.
echo El entrenamiento se detuvo. Puedes cerrar esta ventana.
pause
