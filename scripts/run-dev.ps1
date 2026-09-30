# Starts the recommendation dashboard: model-results FastAPI + Vite.
#
# The API binds to 127.0.0.1 only — it reads data/ (real client names and revenue)
# and must never be reachable from the network. Vite proxies /api to it, so the
# frontend uses relative paths and no confidential data is bundled.
#
# Usage:  .\scripts\run-dev.ps1   then open http://localhost:5173

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path "$root\web\node_modules\.bin\vite.cmd")) {
    Write-Host "Frontend dependencies missing. Installing…" -ForegroundColor Yellow
    # --include=dev is required: this machine has npm config omit=dev set globally,
    # which would otherwise skip vite and typescript entirely.
    Push-Location "$root\web"; npm install --include=dev; Pop-Location
}

Write-Host "API      → http://127.0.0.1:8001/docs" -ForegroundColor Cyan
Write-Host "Dashboard → http://localhost:5173" -ForegroundColor Cyan
Write-Host "Ctrl+C to stop the frontend; the API window closes with it." -ForegroundColor DarkGray

# The API refuses to serve real data/ unauthenticated unless asked in so many words
# (model_results_api/app/config.py). This script is that "so many words": it binds
# 127.0.0.1 explicitly, two lines below, so the data never leaves the machine.
# Without this the API would refuse to start -- and the window is hidden, so the
# refusal would be invisible and the dashboard would just fail to load.
$env:MODEL_RESULTS_ALLOW_ANONYMOUS = "1"

# uvicorn --reload spawns its worker through multiprocessing.spawn, and the PARENT
# creates the listening socket then hands the handle to that child. Stopping only the
# parent PID -- which is what the finally block used to do -- leaves the worker alive,
# still bound to 8001, still serving whatever code was on disk when it launched. Six
# of those accumulated across Ctrl+C runs: every edit appeared to be ignored, --reload
# never fired because its watcher parent was long dead, and a freshly started API
# exited on sight because the port was already held. Refuse to start on top of that
# rather than adding a seventh.
$held = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
if ($held) {
    Write-Host "Le port 8001 est deja occupe." -ForegroundColor Red
    Write-Host "Un serveur precedent tourne encore et repondra a la place du nouveau." -ForegroundColor Red
    Write-Host "Pour le liberer :" -ForegroundColor Yellow
    Write-Host '  Get-CimInstance Win32_Process -Filter "Name=''python.exe''" |' -ForegroundColor Gray
    Write-Host "    Where-Object { `$_.CommandLine -like '*uvicorn*' -or `$_.CommandLine -like '*multiprocessing.spawn*' } |" -ForegroundColor Gray
    Write-Host "    ForEach-Object { taskkill /PID `$_.ProcessId /T /F }" -ForegroundColor Gray
    exit 1
}

$api = Start-Process -FilePath "py" `
    -ArgumentList "-m", "uvicorn", "model_results_api.app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8001" `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

try {
    Push-Location "$root\web"
    npm run dev
} finally {
    Pop-Location
    # /T kills the process tree, not just the parent: see the note above the launch.
    # Stop-Process -Id is what orphaned six workers, so it must not come back here.
    if ($api -and -not $api.HasExited) {
        & taskkill /PID $api.Id /T /F 2>$null | Out-Null
    }
    Write-Host "`nAPI stopped." -ForegroundColor DarkGray
}
