@echo off
setlocal
call "%~dp0ai-fit\Export-Database-Schema.bat"
if errorlevel 1 (
    pause
    exit /b 1
)
start "" notepad.exe "%~dp0ai-fit\AI-FIT_Database_Schema.sql"
