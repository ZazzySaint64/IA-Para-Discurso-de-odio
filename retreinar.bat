@echo off
cd /d "%~dp0"
python retreinar_modelo.py >> historico_execucoes.log 2>&1
