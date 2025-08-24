param(
    [Parameter(Mandatory=$true)]
    [string]$NewImageTag
)

$namespace = "netflix-ml"
$dockerUsername = "mgouse"

Write-Host "=== Blue-Green Deployment ===" -ForegroundColor Green

# Step 1: Deploy to Green
Write-Host "`nDeploying to Green environment..." -ForegroundColor Yellow
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: netflix-api-green
  namespace: $namespace
spec:
  replicas: 2
  selector:
    matchLabels:
      app: netflix-api
      version: green
  template:
    metadata:
      labels:
        app: netflix-api
        version: green
    spec:
      containers:
      - name: netflix-api
        image: $dockerUsername/netflix-api:$NewImageTag
        ports:
        - containerPort: 8000
EOF

# Wait for green deployment
kubectl rollout status deployment/netflix-api-green -n $namespace

# Step 2: Test Green deployment
Write-Host "`nTesting Green deployment..." -ForegroundColor Yellow
# Add your tests here

# Step 3: Switch traffic to Green
Write-Host "`nSwitching traffic to Green..." -ForegroundColor Yellow
kubectl patch service netflix-api-service -n $namespace -p '{"spec":{"selector":{"version":"green"}}}'

# Step 4: Remove Blue deployment
Write-Host "`nRemoving old Blue deployment..." -ForegroundColor Yellow
kubectl delete deployment netflix-api-blue -n $namespace --ignore-not-found=true

# Step 5: Rename Green to Blue for next deployment
kubectl patch deployment netflix-api-green -n $namespace --type='json' -p='[{"op": "replace", "path": "/metadata/name", "value":"netflix-api-blue"}]'

Write-Host "`n✅ Blue-Green deployment complete!" -ForegroundColor Green