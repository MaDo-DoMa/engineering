# One-command reproducibility run: rebuild the stack, retrain every
# instrument with the pinned seeds/versions, restart the serving layer.
#
# Usage (from the engineering directory):
#     .\reproduce.ps1                 # all 5 tickers
#     .\reproduce.ps1 SPY500 GLD      # only the listed tickers
#
# NOTE: the Postgres volume (dashboard price history) is NOT touched -
# it survives the rebuild. To refresh it from Tiingo, use the existing
# sync endpoint of the Java backend afterwards.

$ErrorActionPreference = "Stop"
$etfs = if ($args.Count -gt 0) { $args } else { "NASDAQ", "SPY500", "WIG20", "INVESCO", "GLD" }

Write-Host "== 1/4 Building and starting the stack ==" -ForegroundColor Cyan
docker compose up -d --build

Write-Host "== 2/4 Waiting for the ML service (cold start ~40-60 s) ==" -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep -Seconds 10
    $health = docker compose exec -T python-ml python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=5).status)" 2>$null
} while ($health -ne "200" -and (Get-Date) -lt $deadline)
if ($health -ne "200") { throw "python-ml did not become healthy in time." }
Write-Host "python-ml healthy."

Write-Host "== 3/4 Retraining all instruments (seed 42, pinned versions) ==" -ForegroundColor Cyan
foreach ($t in $etfs) {
    Write-Host "--- $t ---" -ForegroundColor Yellow
    docker compose exec -T -w /app/src -e PYTHONIOENCODING=utf-8 -e OMP_NUM_THREADS=4 `
        python-ml python run_pipeline.py $t
    if ($LASTEXITCODE -ne 0) { throw "Retraining failed for $t (exit code $LASTEXITCODE)." }
}

Write-Host "== 4/4 Restarting the serving layer (clears cached packs) ==" -ForegroundColor Cyan
docker compose restart python-ml
docker compose restart frontend

Write-Host ""
Write-Host "DONE. Open the dashboard (see: .\up.ps1) and verify with:" -ForegroundColor Green
Write-Host "  docker compose exec frontend python /app/test_statistical_tests.py"
