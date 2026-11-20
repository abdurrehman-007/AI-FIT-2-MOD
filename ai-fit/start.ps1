$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Test-VendorUsable {
    $marker = Join-Path $PSScriptRoot '.vendor\flask\__init__.py'
    if (-not (Test-Path $marker)) { return $false }
    try {
        Get-Content -LiteralPath $marker -TotalCount 1 | Out-Null
        return $true
    } catch {
        return $false
    }
}

$python = $null
$taskBundledPython = 'C:\Users\Night-Fury\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if ((Test-VendorUsable) -and (Test-Path $taskBundledPython)) {
    $python = $taskBundledPython
} else {
    if (-not (Test-Path '.venv\Scripts\python.exe')) {
        py -3 -m venv .venv
        & '.venv\Scripts\python.exe' -m pip install -r requirements.txt
    }
    $python = '.venv\Scripts\python.exe'
}

& $python app.py
