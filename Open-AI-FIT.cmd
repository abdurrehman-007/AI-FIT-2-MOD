@echo off
cd /d "%~dp0"
"%~dp0ai-fit\.gpu-runtime\Scripts\python.exe" "%~dp0ai-fit\scripts\run_local.py"
pause
