@echo off
setlocal

set "PYTHON=%~dp0.gpu-runtime\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "DB=%~dp0instance\aifit.sqlite3"
if defined DATA_DIR (
    pushd "%~dp0" || exit /b 1
    for %%I in ("%DATA_DIR%\aifit.sqlite3") do set "DB=%%~fI"
    popd
)
set "OUT=%~dp0AI-FIT_Database_Schema.sql"

if not exist "%DB%" (
    echo SQLite database not found: "%DB%"
    exit /b 1
)

"%PYTHON%" -B -c "import sqlite3, sys; from pathlib import Path; db = sqlite3.connect(Path(sys.argv[1]).resolve().as_uri() + '?mode=ro', uri=True); rows = db.execute(\"SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%%' ORDER BY rowid\").fetchall(); db.close(); assert rows, 'No schema definitions found'; Path(sys.argv[2]).write_text('-- AI-FIT SQLite schema\n\n' + ''.join(sql + ';\n\n' for (sql,) in rows), encoding='utf-8')" "%DB%" "%OUT%"
if errorlevel 1 exit /b 1
echo Schema exported to "%OUT%"
