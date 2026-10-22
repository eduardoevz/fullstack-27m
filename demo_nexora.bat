@echo off
rem Nexora: interfaz web local del modelo v2 final (Fase 14). Abre el navegador en http://127.0.0.1:8765
rem Cierra esta ventana (o pulsa Ctrl+C) para apagar el servidor.
cd /d "%~dp0"
title Nexora - fullstack-27m v2
python -m src.web.server --config config\model_v2_long_c.json --ckpt C:\llm-fullstack-data\checkpoints_v2_long_c_hitos\hito-0018310.pt
pause
