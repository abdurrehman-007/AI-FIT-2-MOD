@echo off
cd /d "%~dp0"
"%~dp0.gpu-runtime\Scripts\python.exe" "%~dp0scripts\run_local.py"
pause
