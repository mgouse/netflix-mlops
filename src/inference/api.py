"""
Netflix Recommendation FastAPI Service
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from fastapi import Response
from datetime import datetime
import os
import psutil
import logging
import json
from typing import List, Optional
import asyncio
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Netflix Recommendation API",
    description="Production-ready recommendation service",
    version="1.0.0"
)

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_errors = Counter('prediction_errors_total', 'Total number of prediction errors')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
active_requests = Gauge('active_requests', 'Number of active requests')

# Global variables for model and data
model = None
user_item_matrix = None
user_encoder = None
movie_encoder = None
production_stats = {
    "predictions": 0,
    "errors": 0,
    "start_time": datetime.now(),
    "last_prediction": None,
    "model_version": "1.0.0"
}

# Production data storage (in production, use database)
production_logs = []

class PredictionRequest(BaseModel):
    user_id: str = Field(..., description="User ID for recommendations")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")

class PredictionResponse(BaseModel):
    user_id: str
    recommendations: List[str]
    scores: List[float]
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    uptime_seconds: float

class ProductionStatsResponse(BaseModel):
    total_predictions: int
    total_errors: int
    error_rate: float
    uptime_seconds: float
    predictions_per_minute: float
    last_prediction: Optional[str]
    model_version: str

@app.on_event("startup")
async def load_model():
    """Load model and encoders on startup"""
    global model, user_item_matrix, user_encoder, movie_encoder
    
    try:
        # Load from local paths or mounted volumes in K8s
        model_path = os.getenv("MODEL_PATH", "D:/Netflix/models/knn_model.pkl")
        data_path = os.getenv("DATA_PATH", "D:/Netflix/data/processed")
        
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Loading data from {data_path}")
        user_item_matrix = pd.read_csv(f"{data_path}/user_item_matrix.csv", index_col=0)
        
        with open(f"{data_path}/user_encoder.pkl", 'rb') as f:
            user_encoder = pickle.load(f)
        
        with open(f"{data_path}/movie_encoder.pkl", 'rb') as f:
            movie_encoder = pickle.load(f)
        
        logger.info("Model and data loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"User-item matrix shape: {user_item_matrix.shape}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("shutdown")
async def save_production_logs():
    """Save production logs on shutdown"""
    try:
        logs_path = os.getenv("LOGS_PATH", "D:/Netflix/logs")
        os.makedirs(logs_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{logs_path}/production_logs_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(production_logs, f, indent=2, default=str)
        
        logger.info(f"Production logs saved to {log_file}")
    except Exception as e:
        logger.error(f"Failed to save production logs: {e}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Netflix Recommendation API",
        "version": "1.0.0",
        "endpoints": [
            "/docs",
            "/health",
            "/readiness",
            "/predict",
            "/production-stats",
            "/download-production-data",
            "/metrics"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for Kubernetes liveness probe"""
    uptime = (datetime.now() - production_stats["start_time"]).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        uptime_seconds=uptime
    )

@app.get("/readiness")
async def readiness():
    """Readiness probe for Kubernetes"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 90 or memory_percent > 90:
        raise HTTPException(
            status_code=503,
            detail=f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%"
        )
    
    return {
        "status": "ready",
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Get movie recommendations for a user"""
    global production_stats
    
    # Track metrics
    active_requests.inc()
    start_time = time.time()
    
    try:
        # Validate user exists
        if request.user_id not in user_encoder.classes_:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        # Encode user ID
        user_idx = user_encoder.transform([request.user_id])[0]
        
        # Get user vector
        user_vector = user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        
        # Get recommendations
        distances, indices = model.kneighbors(
            user_vector,
            n_neighbors=min(request.n_recommendations + 1, len(user_item_matrix))
        )
        
        # Remove self (first neighbor)
        neighbor_indices = indices[0][1:request.n_recommendations + 1]
        neighbor_distances = distances[0][1:request.n_recommendations + 1]
        
        # Get recommended items
        recommendations = []
        scores = []
        
        for idx, distance in zip(neighbor_indices, neighbor_distances):
            # Get top rated movies by similar users
            similar_user_ratings = user_item_matrix.iloc[idx]
            top_movies = similar_user_ratings[similar_user_ratings > 0].nlargest(5)
            
            for movie_idx, rating in top_movies.items():
                movie_id = movie_encoder.inverse_transform([int(movie_idx)])[0]
                score = float(rating * (1 - distance))  # Weight by similarity
                
                if movie_id not in recommendations:
                    recommendations.append(movie_id)
                    scores.append(score)
                
                if len(recommendations) >= request.n_recommendations:
                    break
            
            if len(recommendations) >= request.n_recommendations:
                break
        
        # Update stats
        production_stats["predictions"] += 1
        production_stats["last_prediction"] = datetime.now().isoformat()
        prediction_counter.inc()
        
        # Log prediction for analysis
        prediction_log = {
            "timestamp": datetime.now().isoformat(),
            "user_id": request.user_id,
            "recommendations": recommendations[:request.n_recommendations],
            "scores": scores[:request.n_recommendations]
        }# Save to file for monitoring
        try:
           logs_dir = os.getenv("LOGS_PATH", "/app/logs")
           os.makedirs(logs_dir, exist_ok=True)
           with open(os.path.join(logs_dir, "predictions.jsonl"), "a") as f:
               f.write(json.dumps(prediction_log) + "\n")
        except Exception as e:
           logger.warning(f"Failed to save log: {e}")
           # Add to background task
           background_tasks.add_task(log_prediction, prediction_log)
        
        response = PredictionResponse(
            user_id=request.user_id,
            recommendations=recommendations[:request.n_recommendations],
            scores=scores[:request.n_recommendations],
            model_version=production_stats["model_version"],
            timestamp=datetime.now().isoformat()
        )
        
        # Track latency
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        production_stats["errors"] += 1
        prediction_errors.inc()
        logger.error(f"Prediction error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        active_requests.dec()

async def log_prediction(prediction_data: dict):
    """Background task to log predictions"""
    global production_logs
    production_logs.append(prediction_data)
    
    # Keep only last 10000 predictions in memory
    if len(production_logs) > 10000:
        production_logs = production_logs[-10000:]

@app.get("/production-stats", response_model=ProductionStatsResponse)
async def get_production_stats():
    """Get production statistics"""
    uptime = (datetime.now() - production_stats["start_time"]).total_seconds()
    
    return ProductionStatsResponse(
        total_predictions=production_stats["predictions"],
        total_errors=production_stats["errors"],
        error_rate=production_stats["errors"] / max(production_stats["predictions"], 1),
        uptime_seconds=uptime,
        predictions_per_minute=production_stats["predictions"] / max(uptime / 60, 1),
        last_prediction=production_stats["last_prediction"],
        model_version=production_stats["model_version"]
    )

@app.get("/download-production-data")
async def download_production_data():
    """Download production logs for analysis"""
    if not production_logs:
        raise HTTPException(status_code=404, detail="No production data available")
    
    # Save to temporary file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_file = f"/tmp/production_data_{timestamp}.json"
    
    with open(temp_file, 'w') as f:
        json.dump({
            "export_time": datetime.now().isoformat(),
            "stats": production_stats,
            "predictions": production_logs[-1000:]  # Last 1000 predictions
        }, f, indent=2, default=str)
    
    return FileResponse(
        temp_file,
        media_type="application/json",
        filename=f"netflix_production_data_{timestamp}.json"
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Additional utility endpoints
@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": str(type(model)),
        "n_neighbors": getattr(model, 'n_neighbors', 'N/A'),
        "metric": getattr(model, 'metric', 'N/A'),
        "n_samples_fit": getattr(model, 'n_samples_fit_', 'N/A'),
        "model_version": production_stats["model_version"]
    }

@app.get("/supported-users")
async def supported_users():
    """Get list of supported user IDs (first 100 for demo)"""
    if user_encoder is None:
        raise HTTPException(status_code=503, detail="Encoder not loaded")
    
    return {
        "total_users": len(user_encoder.classes_),
        "sample_users": user_encoder.classes_[:100].tolist()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc.detail), "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "path": str(request.url)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)