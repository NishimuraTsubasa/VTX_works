@echo off
setlocal
cd /d %~dp0
py scripts\check_pdf_font.py --config config\model_config.py
pause
