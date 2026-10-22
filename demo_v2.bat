@echo off
rem Demo del modelo v2 final (Fase 11): continua tu codigo o rellena el hueco donde pongas <CURSOR>.
rem Escribe codigo y termina con una linea que solo tenga un punto (.)
cd /d "%~dp0"
title Demo fullstack-27m v2
python src\cli.py --config config\model_v2_long_c.json --ckpt C:\llm-fullstack-data\checkpoints_v2_long_c_hitos\hito-0018310.pt
pause
