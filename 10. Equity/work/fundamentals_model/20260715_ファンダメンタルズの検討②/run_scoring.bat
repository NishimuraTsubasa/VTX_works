@echo off
cd /d %~dp0
python scripts\run_pipeline.py --config config\model_config.py
pause
