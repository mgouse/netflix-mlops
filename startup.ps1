# Daily startup script
Write-Host "Starting Netflix MLOps Environment..." -ForegroundColor Green

# Start Minikube
minikube start --driver=docker --cpus=2 --memory=5120

# Activate Python environment
cd D:\Netflix
.\venv\Scripts\activate

# Check cluster status
kubectl get nodes
kubectl get pods --all-namespaces

Write-Host "Environment ready!" -ForegroundColor Green