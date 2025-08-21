# Phase 2 Verification Script
Write-Host "=== Phase 2 Verification ===" -ForegroundColor Green

# 1. Check PostgreSQL
Write-Host "`n1. PostgreSQL Status:" -ForegroundColor Yellow
$pgService = Get-Service -Name "postgresql-x64-14" -ErrorAction SilentlyContinue
if ($pgService.Status -eq 'Running') {
    Write-Host "   ✓ PostgreSQL is running" -ForegroundColor Green
} else {
    Write-Host "   ✗ PostgreSQL is not running" -ForegroundColor Red
}

# 2. Check MLflow database
Write-Host "`n2. MLflow Database:" -ForegroundColor Yellow
$dbCheck = psql -U postgres -d mlflow_db -c "\dt" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "   ✓ MLflow database accessible" -ForegroundColor Green
    $tableCount = ($dbCheck | Select-String "rows" | Measure-Object).Count
    Write-Host "   ✓ Found $tableCount MLflow tables" -ForegroundColor Green
} else {
    Write-Host "   ✗ Cannot access MLflow database" -ForegroundColor Red
}

# 3. Check environment variables
Write-Host "`n3. Environment Variables:" -ForegroundColor Yellow

# Ensure virtual environment is activated and .env is loaded before Python call
cd D:\Netflix
& .\venv\Scripts\activate

# Use single quotes for the PowerShell string, and double quotes inside for Python
$trackingUri = python -c 'from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv("MLFLOW_TRACKING_URI", "NOT SET"))'

Write-Host "   MLflow Tracking URI: $trackingUri" -ForegroundColor Cyan

# 4. Check models
Write-Host "`n4. Trained Models:" -ForegroundColor Yellow
$models = Get-ChildItem D:\Netflix\models\*.pkl -ErrorAction SilentlyContinue
if ($models) {
    Write-Host "   ✓ Found $($models.Count) model files:" -ForegroundColor Green
    $models | ForEach-Object { Write-Host "     - $($_.Name)" -ForegroundColor Gray }
} else {
    Write-Host "   ✗ No model files found" -ForegroundColor Red
}

# 5. Check hyperparameter results
Write-Host "`n5. Hyperparameter Results:" -ForegroundColor Yellow
if (Test-Path "D:\Netflix\models\hyperparameter_results.csv") {
    Write-Host "   ✓ Hyperparameter results found" -ForegroundColor Green
    $results = Import-Csv "D:\Netflix\models\hyperparameter_results.csv"
    $results | Format-Table -AutoSize
} else {
    Write-Host "   ✗ No hyperparameter results found" -ForegroundColor Red
}

# 6. Check validation report
Write-Host "`n6. Validation Report:" -ForegroundColor Yellow
if (Test-Path "D:\Netflix\models\validation_report.json") {
    Write-Host "   ✓ Validation report found" -ForegroundColor Green
    Get-Content "D:\Netflix\models\validation_report.json" | ConvertFrom-Json | ConvertTo-Json -Depth 10
} else {
    Write-Host "   ✗ No validation report found" -ForegroundColor Red
}

# 7. Check final model
Write-Host "`n7. Final Model:" -ForegroundColor Yellow
if (Test-Path "D:\Netflix\models\knn_model.pkl") {
    Write-Host "   ✓ Final model (knn_model.pkl) exists" -ForegroundColor Green
} else {
    Write-Host "   ✗ Final model not found" -ForegroundColor Red
}

# 8. Check MLflow artifacts
Write-Host "`n8. MLflow Artifacts:" -ForegroundColor Yellow
$artifactDirs = @("D:\Netflix\mlflow_experiments", "D:\Netflix\mlruns")
foreach ($dir in $artifactDirs) {
    if (Test-Path $dir) {
        $count = (Get-ChildItem $dir -Recurse -File | Measure-Object).Count
        Write-Host "   ✓ $dir exists with $count files" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $dir not found" -ForegroundColor Red
    }

Write-Host "`n=== Verification Complete ===" -ForegroundColor Green
