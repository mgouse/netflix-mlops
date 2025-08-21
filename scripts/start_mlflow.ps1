# Load environment variables and start MLflow UI
$envFile = "D:\Netflix\.env"

if (Test-Path $envFile) {
    # Read .env file
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)\s*=\s*(.+)\s*$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Item -Path "env:$name" -Value $value
        }
    }
    
    Write-Host "Environment variables loaded from .env" -ForegroundColor Green
    
    # Get environment variables
    $backendUri = $env:MLFLOW_BACKEND_STORE_URI
    $artifactRoot = $env:MLFLOW_ARTIFACT_ROOT
    $port = $env:MLFLOW_PORT
    
    Write-Host "Starting MLflow UI..." -ForegroundColor Yellow
    Write-Host "Backend URI: $backendUri" -ForegroundColor Cyan
    Write-Host "Artifact Root: $artifactRoot" -ForegroundColor Cyan
    Write-Host "Port: $port" -ForegroundColor Cyan
    
    # Start MLflow
    mlflow ui --backend-store-uri "$backendUri" --default-artifact-root "$artifactRoot" --port $port
} else {
    Write-Host "Error: .env file not found at $envFile" -ForegroundColor Red
}