# Fix directory issues
Write-Host "Fixing directory structure..." -ForegroundColor Yellow

# Create correct artifact directory
$artifactDir = "D:\Netflix\mlflow_experiments"
if (!(Test-Path $artifactDir)) {
    New-Item -ItemType Directory -Path $artifactDir -Force
    Write-Host "Created: $artifactDir" -ForegroundColor Green
}

# Create models directory
$modelsDir = "D:\Netflix\models"
if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force
    Write-Host "Created: $modelsDir" -ForegroundColor Green
}

Write-Host "Directory structure fixed!" -ForegroundColor Green