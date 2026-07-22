@echo off
cd /d %~dp0
py -m pip install -e .
py scripts\run_pipeline.py --config config\model_config.py
pause
