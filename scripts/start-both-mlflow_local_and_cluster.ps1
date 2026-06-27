# start-both-mlflow.ps1
Write-Host "Starting both MLflow instances..."
Write-Host "1. Starting local MLflow 3.2.0 with PostgreSQL on port 5001..."
Start-Process powershell -ArgumentList "-Command", "mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri postgresql://postgres:postgres@localhost:5432/mlflow_db --default-artifact-root file:///D:/Netflix/mlflow_experiments"

Write-Host "2. Port-forwarding cluster MLflow to port 5000..."
kubectl port-forward -n argo svc/mlflow-service 5000:5000

Write-Host "Both instances started:"
Write-Host "- Local MLflow 3.2.0 (PostgreSQL): http://localhost:5001"
Write-Host "- Cluster MLflow 2.10.0: http://localhost:5000"
