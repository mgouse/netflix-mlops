"""
Validate environment configuration
"""
import os
from dotenv import load_dotenv
from pathlib import Path
import psycopg2
import sys

# Load environment variables
env_path = Path('D:/Netflix/.env')
load_dotenv(dotenv_path=env_path)

def validate_env():
    """Validate all required environment variables"""
    required_vars = [
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_HOST',
        'POSTGRES_PORT',
        'POSTGRES_DB',
        'MLFLOW_BACKEND_STORE_URI',
        'MLFLOW_ARTIFACT_ROOT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_postgres_connection():
    """Test PostgreSQL connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST'),
            port=os.getenv('POSTGRES_PORT'),
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

def validate_paths():
    """Validate required paths exist"""
    artifact_root = os.getenv('MLFLOW_ARTIFACT_ROOT', '').replace('file:///', '')
    
    if not os.path.exists(artifact_root):
        os.makedirs(artifact_root, exist_ok=True)
        print(f"✅ Created artifact directory: {artifact_root}")
    else:
        print(f"✅ Artifact directory exists: {artifact_root}")
    
    return True

if __name__ == "__main__":
    print("Validating environment configuration...\n")
    
    if not env_path.exists():
        print(f"❌ .env file not found at {env_path}")
        print("Please create .env file from .env.example")
        sys.exit(1)
    
    all_valid = True
    all_valid &= validate_env()
    all_valid &= test_postgres_connection()
    all_valid &= validate_paths()
    
    if all_valid:
        print("\n✅ Environment validation successful!")
    else:
        print("\n❌ Environment validation failed. Please fix the issues above.")
        sys.exit(1)