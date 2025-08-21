"""
Validate and test recommendation model
"""
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import logging
from datetime import datetime
import shutil

# Add parent directory to path
sys.path.append('D:/Netflix/src')
from utils.mlflow_setup import setup_mlflow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_best_model():
    """Load the best model based on hyperparameter results"""
    # Load hyperparameter results
    results_path = 'D:/Netflix/models/hyperparameter_results.csv'
    if not os.path.exists(results_path):
        logger.error("No hyperparameter results found. Train models first.")
        return None, None
    
    results_df = pd.read_csv(results_path)
    best_model_idx = results_df['avg_distance'].idxmin()
    best_params = results_df.iloc[best_model_idx]
    
    logger.info(f"Best model parameters: n_neighbors={best_params['n_neighbors']}, metric={best_params['metric']}")
    
    # Load the best model
    model_path = f"D:/Netflix/models/knn_model_{int(best_params['n_neighbors'])}_{best_params['metric']}.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model, best_params

def validate_model():
    """Validate model performance"""
    logger.info("Starting model validation...")
    
    # Setup MLflow
    mlflow_client = setup_mlflow()
    
    # Load best model
    model, best_params = load_best_model()
    if model is None:
        return
    
    # Load data
    user_item_matrix = pd.read_csv('D:/Netflix/data/processed/user_item_matrix.csv', index_col=0)
    logger.info(f"Loaded data: {user_item_matrix.shape}")
    
    # Test recommendations for sample users
    n_test_users = 5
    test_user_indices = np.random.choice(len(user_item_matrix), n_test_users, replace=False)
    
    logger.info("\nSample Recommendations:")
    for i, user_idx in enumerate(test_user_indices):
        user_vector = user_item_matrix.iloc[user_idx].values.reshape(1, -1)
        distances, indices = model.kneighbors(user_vector, n_neighbors=10)
        
        logger.info(f"\nUser {user_idx}:")
        logger.info(f"  Nearest neighbors: {indices[0][:5]}")
        logger.info(f"  Distances: {distances[0][:5]}")
    
    # Calculate overall metrics
    coverage = len(np.unique(model._fit_X)) / len(user_item_matrix.columns)
    logger.info(f"\nCatalog coverage: {coverage:.2%}")
    
    # Copy best model to standard location
    best_model_path = f"D:/Netflix/models/knn_model_{int(best_params['n_neighbors'])}_{best_params['metric']}.pkl"
    final_model_path = 'D:/Netflix/models/knn_model.pkl'
    
    shutil.copy(best_model_path, final_model_path)
    logger.info(f"Best model copied to: {final_model_path}")
    
    # Save validation report
    validation_report = {
        'timestamp': datetime.now().isoformat(),
        'best_model': {
            'n_neighbors': int(best_params['n_neighbors']),
            'metric': best_params['metric'],
            'avg_distance': float(best_params['avg_distance'])
        },
        'validation_metrics': {
            'catalog_coverage': coverage,
            'n_test_users': n_test_users
        }
    }
    
    report_path = 'D:/Netflix/models/validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Validation report saved to: {report_path}")
    
    return validation_report

if __name__ == "__main__":
    # Check if models exist
    if not os.path.exists('D:/Netflix/models'):
        logger.error("No models found. Please run training first.")
        sys.exit(1)
    
    # Run validation
    validate_model()