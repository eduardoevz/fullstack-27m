@echo off
cd /d "%~dp0"
python src\estado.py --config config\model_v2.json --log logs\train_v2.csv
pause
