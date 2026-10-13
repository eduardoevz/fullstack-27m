@echo off
rem Fase 11: prueba de FIM a nivel de texto desde el paso 7.500 (aborta en el 9.000 si no mejora). No toca otros checkpoints.
rem Para parar sin perder nada: ejecuta parar.bat (o pulsa Ctrl+C aqui). Cierra la ventana solo tras "Checkpoint guardado".
cd /d "%~dp0"
title Entrenamiento LLM Full Stack - Fase 11 (FIM en texto)
if exist "C:\llm-fullstack-data\checkpoints_v2_text\ckpt-*.pt" (
    echo Reanudando el entrenamiento v2 desde el ultimo checkpoint...
    python src\train.py --config config\model_v2_text.json --log logs\train_v2_text.csv --resume
) else (
    echo Empezando el entrenamiento v2 desde el checkpoint de la prueba FIM...
    python src\train.py --config config\model_v2_text.json --log logs\train_v2_text.csv
)
echo.
echo El entrenamiento se detuvo. Puedes cerrar esta ventana.
pause
