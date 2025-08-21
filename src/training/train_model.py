"""
Train KNN recommendation model with MLflow tracking
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
import mlflow.sklearn
import pickle
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append('D:/Netflix/src')
from utils.mlflow_setup import setup_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(model, test_data, train_data, k=10):
    """Calculate recommendation metrics"""
    # Ensure k doesn't exceed the number of training samples
    k = min(k, len(train_data) - 1)
    
    # Get recommendations for all test users
    distances, indices = model.kneighbors(test_data, n_neighbors=k)
    
    # Calculate average distance (lower is better)
    avg_distance = np.mean(distances)
    
    # Calculate coverage (what % of items get recommended)
    unique_recommendations = np.unique(indices.flatten())
    coverage = len(unique_recommendations) / len(train_data)
    
    # Calculate diversity (average distance between recommended items)
    diversity_scores = []
    for user_recs in indices:
        if len(user_recs) > 1:
            # Use train_data instead of test_data for getting recommended items
            user_items = train_data[user_recs]
            # Calculate pairwise distances
            item_distances = []
            for i in range(len(user_items)):
                for j in range(i+1, len(user_items)):
                    dist = np.linalg.norm(user_items[i] - user_items[j])
                    item_distances.append(dist)
            if item_distances:
                diversity_scores.append(np.mean(item_distances))
    
    diversity = np.mean(diversity_scores) if diversity_scores else 0
    
    return {
        'avg_distance': avg_distance,
        'coverage': coverage,
        'diversity': diversity
    }

def train_knn_model(n_neighbors=10, metric='cosine', algorithm='brute'):
    """Train KNN collaborative filtering model"""
    logger.info(f"Starting training with n_neighbors={n_neighbors}, metric={metric}")
    
    # Setup MLflow
    mlflow_client = setup_mlflow()
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"knn_{n_neighbors}_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        try:
            # Log parameters
            mlflow.log_param("model_type", "KNN")
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("metric", metric)
            mlflow.log_param("algorithm", algorithm)
            
            # Load processed data
            logger.info("Loading processed data...")
            user_item_matrix = pd.read_csv('D:/Netflix/data/processed/user_item_matrix.csv', index_col=0)
            logger.info(f"Data shape: {user_item_matrix.shape}")
            
            # Log data characteristics
            mlflow.log_param("n_users", user_item_matrix.shape[0])
            mlflow.log_param("n_items", user_item_matrix.shape[1])
            sparsity = 1 - (user_item_matrix != 0).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
            mlflow.log_param("sparsity", round(sparsity, 4))
            
            # Split data for validation
            train_data, test_data = train_test_split(
                user_item_matrix.values, 
                test_size=0.2, 
                random_state=42
            )
            logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            
            # Ensure n_neighbors is valid
            n_neighbors_actual = min(n_neighbors, len(train_data) - 1)
            if n_neighbors_actual != n_neighbors:
                logger.warning(f"Adjusted n_neighbors from {n_neighbors} to {n_neighbors_actual} due to training data size")
                mlflow.log_param("n_neighbors_actual", n_neighbors_actual)
            
            # Train KNN model
            logger.info("Training KNN model...")
            model = NearestNeighbors(
                n_neighbors=n_neighbors_actual,
                metric=metric,
                algorithm=algorithm,
                n_jobs=-1  # Use all CPU cores
            )
            
            # Fit model
            import time
            start_time = time.time()
            model.fit(train_data)
            training_time = time.time() - start_time
            
            logger.info(f"Model trained in {training_time:.2f} seconds")
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics = calculate_metrics(model, test_data, train_data, k=n_neighbors_actual)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                logger.info(f"{metric_name}: {metric_value:.4f}")
            
            # Save model locally
            model_dir = "D:/Netflix/models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_filename = f"knn_model_{n_neighbors}_{metric}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {model_path}")
            
            # Log model to MLflow
            logger.info("Logging model to MLflow...")
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="netflix_knn_recommender",
                pip_requirements=["scikit-learn==1.3.0", "numpy==1.24.3", "pandas==1.5.3"]
            )
            
            # Log additional artifacts
            if os.path.exists("D:/Netflix/data/processed/user_encoder.pkl"):
                mlflow.log_artifact("D:/Netflix/data/processed/user_encoder.pkl")
            if os.path.exists("D:/Netflix/data/processed/movie_encoder.pkl"):
                mlflow.log_artifact("D:/Netflix/data/processed/movie_encoder.pkl")
            mlflow.log_artifact(model_path)
            
            # Create and log model info
            model_info = {
                "model_type": "KNN",
                "n_neighbors": n_neighbors,
                "n_neighbors_actual": n_neighbors_actual,
                "metric": metric,
                "training_samples": len(train_data),
                "test_samples": len(test_data),
                "metrics": metrics,
                "training_time": training_time,
                "model_path": model_path
            }
            
            # Save model info
            import json
            info_path = os.path.join(model_dir, f"model_info_{n_neighbors}_{metric}.json")
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact(info_path)
            
            logger.info("Training completed successfully!")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            mlflow.log_param("error", str(e))
            raise

def run_hyperparameter_search():
    """Run hyperparameter search"""
    logger.info("Starting hyperparameter search...")
    
    # Define hyperparameter grid
    n_neighbors_list = [5, 10, 20]
    metric_list = ['cosine', 'euclidean']
    
    results = []
    
    for n_neighbors in n_neighbors_list:
        for metric in metric_list:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training with n_neighbors={n_neighbors}, metric={metric}")
                logger.info(f"{'='*50}")
                
                model, metrics = train_knn_model(n_neighbors, metric)
                
                results.append({
                    'n_neighbors': n_neighbors,
                    'metric': metric,
                    'avg_distance': metrics['avg_distance'],
                    'coverage': metrics['coverage'],
                    'diversity': metrics['diversity']
                })
                
            except Exception as e:
                logger.error(f"Error training model: {e}")
                continue
    
    # Save results summary
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs('D:/Netflix/models', exist_ok=True)
        results_df.to_csv('D:/Netflix/models/hyperparameter_results.csv', index=False)
        logger.info("\nHyperparameter search results:")
        logger.info(results_df.to_string())
        
        # Find best model
        best_model = results_df.loc[results_df['avg_distance'].idxmin()]
        logger.info(f"\nBest model: n_neighbors={best_model['n_neighbors']}, metric={best_model['metric']}")

if __name__ == "__main__":
    # Check if data exists
    if not os.path.exists('D:/Netflix/data/processed/user_item_matrix.csv'):
        logger.error("Processed data not found. Please run preprocessing first.")
        sys.exit(1)
    
    # Run hyperparameter search
    run_hyperparameter_search()