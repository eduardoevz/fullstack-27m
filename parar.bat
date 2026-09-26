@echo off
rem Pide al entrenamiento que guarde y se detenga (tarda hasta ~30 s, lo que dura un paso).
cd /d "%~dp0"
type nul > PARAR.txt
echo Senal enviada. Espera a que la ventana del entrenamiento diga "Checkpoint guardado".
pause
