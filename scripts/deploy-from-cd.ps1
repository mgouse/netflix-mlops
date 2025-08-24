param(
    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest"
)

Write-Host "=== Netflix API Deployment Script ===" -ForegroundColor Green

# Configuration
$dockerUsername = "mgouse"  # Replace with your Docker Hub username
$imageName = "$dockerUsername/netflix-api:$ImageTag"
$namespace = "netflix-ml"

# Check if Minikube is running
Write-Host "`nChecking Minikube status..." -ForegroundColor Yellow
$minikubeStatus = minikube status --format='{{.Host}}'
if ($minikubeStatus -ne "Running") {
    Write-Host "Starting Minikube..." -ForegroundColor Yellow
    minikube start --driver=docker --cpus=2 --memory=5120
}

# Pull latest image from Docker Hub
Write-Host "`nPulling image from Docker Hub..." -ForegroundColor Yellow
docker pull $imageName

# Load image into Minikube
Write-Host "`nLoading image into Minikube..." -ForegroundColor Yellow
minikube image load $imageName

# Update Kubernetes deployment
Write-Host "`nUpdating Kubernetes deployment..." -ForegroundColor Yellow
kubectl set image deployment/netflix-api netflix-api=$imageName -n $namespace

# Wait for rollout
Write-Host "`nWaiting for rollout to complete..." -ForegroundColor Yellow
kubectl rollout status deployment/netflix-api -n $namespace

# Show deployment status
Write-Host "`nDeployment Status:" -ForegroundColor Green
kubectl get deployment netflix-api -n $namespace
kubectl get pods -n $namespace -l app=netflix-api

# Get service URL
Write-Host "`nService URL:" -ForegroundColor Green
$serviceUrl = minikube service netflix-api-service -n $namespace --url
Write-Host $serviceUrl

# Test the deployment
Write-Host "`nTesting deployment..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
try {
    $response = Invoke-WebRequest -Uri "$serviceUrl/health" -UseBasicParsing
    Write-Host "✅ Health check passed!" -ForegroundColor Green
    $response.Content
} catch {
    Write-Host "❌ Health check failed!" -ForegroundColor Red
}