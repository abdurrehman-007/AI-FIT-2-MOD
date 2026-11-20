@echo off
setlocal
set "SCHEMA=%~dp0AI-FIT-Feature-Schema.html"
if not exist "%SCHEMA%" (
    echo Schema file not found: "%SCHEMA%"
    pause
    exit /b 1
)
start "" "%SCHEMA%"
