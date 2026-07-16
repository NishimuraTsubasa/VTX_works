@echo off
cd /d %~dp0
python scripts\run_binscatter.py --config config\model_config.py
pause
