param(
    [string]$ExistingProject = 'E:\Comsats\8th Semester\Fyp 2\Virtual-Try-On-main'
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'ai-fit\.gpu-runtime\Scripts\python.exe')) {
    & (Join-Path $PSScriptRoot 'ai-fit\.gpu-runtime\Scripts\python.exe') (Join-Path $PSScriptRoot 'ai-fit\scripts\run_local.py')
    exit $LASTEXITCODE
}
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Path 'instance' -Force | Out-Null
$taskGpuPython = Join-Path $ExistingProject '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskGpuPython)) { throw "Existing GPU environment not found: $taskGpuPython" }
$env:AIFIT_EXISTING_PROJECT = $ExistingProject
if ($env:AIFIT_ENGINE -ne 'catvton' -and -not (Test-Path -LiteralPath '.models/deco/unet/diffusion_pytorch_model.safetensors')) {
    Write-Host 'Downloading the try-on model (3.4 GB). Interrupted downloads can resume.'
    & $taskGpuPython -B gpu_service/download_deco.py
    if ($LASTEXITCODE -ne 0) { throw 'Model setup failed. Run this command again to resume.' }
}
$env:PYTHONPATH = @(
    (Join-Path $PSScriptRoot '.preprocess-deps'),
    (Join-Path $PSScriptRoot 'gpu_service\vendor'),
    (Join-Path $PSScriptRoot 'gpu_service')
) -join ';'
& $taskGpuPython -B -c "import onnxruntime"
if ($LASTEXITCODE -ne 0) {
    & $taskGpuPython -m pip install --target .preprocess-deps --upgrade --no-deps onnxruntime==1.20.1
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the local CPU segmentation runtime.' }
}
& $taskGpuPython -B gpu_service/setup_preprocessing.py
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing model setup failed. Check your internet connection and retry.' }
& $taskGpuPython -B -c "from preprocessing import PhotoPreprocessor; p=PhotoPreprocessor('.models/preprocessing'); p.close()"
if ($LASTEXITCODE -ne 0) { throw 'ONNX/MediaPipe preprocessing initialization failed. Inspect the error above.' }
& $taskGpuPython -B -c "import importlib.util,sys; sys.path.insert(0,'.localdeps'); sys.exit(0 if importlib.util.find_spec('qrcode') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $taskGpuPython -m pip install --target .localdeps --no-deps qrcode==8.2
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the QR dependency.' }
}
$taskGpuProcess = $null
try {
    $taskReady = $false
    try { $taskHealth = Invoke-RestMethod 'http://127.0.0.1:8001/health' -TimeoutSec 3; $taskReady = $taskHealth.ready -and $taskHealth.pose_validation -and $taskHealth.garment_segmentation } catch {}
    if (-not $taskReady) {
        $taskGpuProcess = Start-Process -FilePath $taskGpuPython -ArgumentList @('-B', '-u', 'gpu_service/local_windows.py') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput 'instance/gpu-output.log' -RedirectStandardError 'instance/gpu-error.log'
        Write-Host 'Loading the try-on model on your GPU. Logs: instance/gpu-output.log and gpu-error.log'
        for ($taskAttempt = 0; $taskAttempt -lt 90; $taskAttempt++) {
            if ($taskGpuProcess.HasExited) { throw 'GPU service stopped. Check instance/gpu-error.log.' }
            try { $taskHealth = Invoke-RestMethod 'http://127.0.0.1:8001/health' -TimeoutSec 2; $taskReady = $taskHealth.ready -and $taskHealth.pose_validation -and $taskHealth.garment_segmentation } catch {}
            if ($taskReady) { break }
            Start-Sleep -Seconds 2
        }
        if (-not $taskReady) { throw 'GPU startup timed out. Check instance/gpu-error.log.' }
    }
    $env:CATVTON_URL = 'http://127.0.0.1:8001'
    $env:CATVTON_API_KEY = (Get-Content -LiteralPath 'instance/gpu.key' -Raw).Trim()
    Write-Host 'GPU ready. Open http://127.0.0.1:5000. Press Ctrl+C to stop.'
    & $taskGpuPython -B app.py
} finally {
    if ($taskGpuProcess -and -not $taskGpuProcess.HasExited) { Stop-Process -Id $taskGpuProcess.Id }
}
