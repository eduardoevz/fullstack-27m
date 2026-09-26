@echo off
rem Inicia o reanuda el entrenamiento. Cierra esta ventana solo despues de que diga "Checkpoint guardado".
rem Para parar sin perder nada: ejecuta parar.bat (o pulsa Ctrl+C aqui).
cd /d "%~dp0"
title Entrenamiento LLM Full Stack
if exist "C:\llm-fullstack-data\checkpoints\ckpt-*.pt" (
    echo Reanudando desde el ultimo checkpoint...
    python src\train.py --resume
) else (
    echo Primera sesion: empezando desde cero...
    python src\train.py
)
echo.
echo El entrenamiento se detuvo. Puedes cerrar esta ventana.
pause
