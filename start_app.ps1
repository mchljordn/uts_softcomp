$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host "Virtual environment belum ada di .venv" -ForegroundColor Yellow
    Write-Host "Jalankan dulu: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

Write-Host "Menjalankan Streamlit dengan interpreter project:" -ForegroundColor Cyan
Write-Host $pythonExe -ForegroundColor Cyan

& $pythonExe -m streamlit run app-copy.py
