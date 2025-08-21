"""
MLflow setup and configuration with environment variables
"""
import mlflow
import os
from sqlalchemy import create_engine
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('D:/Netflix/.env')
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_uri():
    """Construct database URI from environment variables"""
    user = os.getenv('POSTGRES_USER', 'postgres')
    password = os.getenv('POSTGRES_PASSWORD', 'password')
    host = os.getenv('POSTGRES_HOST', 'localhost')
    port = os.getenv('POSTGRES_PORT', '5432')
    database = os.getenv('POSTGRES_DB', 'mlflow_db')
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

def setup_mlflow():
    """Configure MLflow with PostgreSQL backend using environment variables"""
    # Get configuration from environment
    db_uri = os.getenv('MLFLOW_BACKEND_STORE_URI', get_db_uri())
    artifact_root = os.getenv('MLFLOW_ARTIFACT_ROOT', 'file:///D:/Netflix/mlflow_experiments')
    
    # IMPORTANT: Set both tracking URI and registry URI
    mlflow.set_tracking_uri(db_uri)
    mlflow.set_registry_uri(db_uri)
    
    # Also set as environment variable for child processes
    os.environ['MLFLOW_TRACKING_URI'] = db_uri
    
    # Create experiments directory if not exists
    artifact_location = artifact_root.replace('file:///', '')
    os.makedirs(artifact_location, exist_ok=True)
    
    # Create or get experiment
    experiment_name = "netflix_recommendation"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_root
            )
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        raise
    
    # Set as active experiment
    mlflow.set_experiment(experiment_name)
    
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow registry URI: {mlflow.get_registry_uri()}")
    logger.info(f"Active experiment: {experiment_name}")
    logger.info(f"Artifact location: {artifact_root}")
    
    return mlflow

def test_connection():
    """Test MLflow connection"""
    try:
        mlflow_client = setup_mlflow()
        # Try to log a test parameter
        with mlflow.start_run():
            mlflow.log_param("test", "connection")
            mlflow.log_param("environment", os.getenv('ENVIRONMENT', 'development'))
            mlflow.end_run()
        logger.info("MLflow connection successful!")
        logger.info(f"Current tracking URI: {mlflow.get_tracking_uri()}")
        return True
    except Exception as e:
        logger.error(f"MLflow connection failed: {e}")
        logger.error("Check your .env file and PostgreSQL connection")
        return False

if __name__ == "__main__":
    # Check if .env exists
    if not env_path.exists():
        logger.error(f".env file not found at {env_path}")
        logger.error("Please create .env file from .env.example")
        exit(1)
    
    test_connection()