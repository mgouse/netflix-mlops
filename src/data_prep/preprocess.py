"""
Preprocess Netflix data for ML training
Compatible with scikit-learn 1.6.1
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess_data():
    """Load and preprocess Netflix data"""
    print("Loading raw data...")
    
    # Load data
    movies_df = pd.read_csv('D:/Netflix/data/raw/movies.csv')
    ratings_df = pd.read_csv('D:/Netflix/data/raw/ratings.csv')
    
    print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
    
    # Merge datasets
    data = ratings_df.merge(movies_df, on='movie_id')
    
    # Encode categorical variables
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    data['user_id_encoded'] = user_encoder.fit_transform(data['user_id'])
    data['movie_id_encoded'] = movie_encoder.fit_transform(data['movie_id'])
    
    # Create user-item matrix for collaborative filtering
    user_item_matrix = data.pivot_table(
        index='user_id_encoded',
        columns='movie_id_encoded',
        values='rating',
        fill_value=0
    )
    
    # Save processed data
    os.makedirs('D:/Netflix/data/processed', exist_ok=True)
    
    # Save with explicit index handling for pandas 2.2.3
    data.to_csv('D:/Netflix/data/processed/ratings_processed.csv', index=False)
    user_item_matrix.to_csv('D:/Netflix/data/processed/user_item_matrix.csv')
    
    # Save encoders for inference
    with open('D:/Netflix/data/processed/user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open('D:/Netflix/data/processed/movie_encoder.pkl', 'wb') as f:
        pickle.dump(movie_encoder, f)
    
    # Save metadata
    metadata = {
        'n_users': len(user_encoder.classes_),
        'n_movies': len(movie_encoder.classes_),
        'n_ratings': len(data),
        'matrix_shape': user_item_matrix.shape
    }
    
    with open('D:/Netflix/data/processed/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Processed data shape: {data.shape}")
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    print("Saved all processed files to D:/Netflix/data/processed/")
    
    return data, user_item_matrix

if __name__ == "__main__":
    preprocess_data()