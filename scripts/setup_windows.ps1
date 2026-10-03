param([switch]$SkipModel)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

function Invoke-Python {
    & $pythonPath @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.11 -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Install CPython 3.11 x64 before running setup.' }
}
Invoke-Python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11 for this Windows environment.'"
Invoke-Python -m pip install --disable-pip-version-check pip==25.0.1 setuptools==77.0.3 wheel==0.45.1
Invoke-Python -m pip install --disable-pip-version-check torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
Invoke-Python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot 'requirements-windows.lock')
Invoke-Python -m pip install --disable-pip-version-check --no-deps -e $projectRoot
Invoke-Python -m pip check
if ($SkipModel) {
    Invoke-Python -m open_video_summary prepare-demo --without-model
} else {
    Invoke-Python -m open_video_summary prepare-demo
    Invoke-Python -m open_video_summary doctor
}
Write-Host 'Setup completed. Run .\.venv\Scripts\python.exe -m open_video_summary summarize from the repository.'
