# Kubernetes Deployment Health Check Script
Write-Host "=== Netflix API Deployment Status ===" -ForegroundColor Green

# Check namespace
Write-Host "`nNamespace:" -ForegroundColor Yellow
kubectl get namespace netflix-ml

# Check pods
Write-Host "`nPods:" -ForegroundColor Yellow
kubectl get pods -n netflix-ml -o wide

# Check service
Write-Host "`nService:" -ForegroundColor Yellow
kubectl get svc -n netflix-ml

# Check HPA
Write-Host "`nHorizontal Pod Autoscaler:" -ForegroundColor Yellow
kubectl get hpa -n netflix-ml

# Check endpoints
Write-Host "`nEndpoints:" -ForegroundColor Yellow
kubectl get endpoints -n netflix-ml

# Get service URL
$SERVICE_URL = minikube service netflix-api-service -n netflix-ml --url
Write-Host "`nService URL: $SERVICE_URL" -ForegroundColor Cyan

# Test health endpoint
Write-Host "`nHealth Check:" -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing
    $health.Content | ConvertFrom-Json | ConvertTo-Json
} catch {
    Write-Host "Health check failed: $_" -ForegroundColor Red
}

# Show recent logs
Write-Host "`nRecent Logs:" -ForegroundColor Yellow
kubectl logs -l app=netflix-api -n netflix-ml --tail=10