@echo off
rem Fase 11 (corrida c): v2 largo reanudado desde el paso 10.000 con peso de parada 8. Aborta si falla la regla de FIM (ver config).
rem Guarda un hito cada 1.000 pasos en C:\llm-fullstack-data\checkpoints_v2_long_c_hitos (nunca se borran).
rem Para parar sin perder nada: ejecuta parar.bat (o pulsa Ctrl+C aqui). Cierra la ventana solo tras "Checkpoint guardado".
cd /d "%~dp0"
title Entrenamiento LLM Full Stack - Fase 11 (v2 largo, corrida c)
echo Reanudando el entrenamiento v2 (corrida c) desde el ultimo checkpoint...
python src\train.py --config config\model_v2_long_c.json --log logs\train_v2_long_c.csv --resume
echo.
echo El entrenamiento se detuvo. Puedes cerrar esta ventana.
pause
