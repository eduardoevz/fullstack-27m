@echo off
cd /d "%~dp0"
python src\estado.py --config config\model_v2_long.json --log logs\train_v2_long.csv
pause
