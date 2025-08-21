"""
Generate synthetic Netflix movie ratings data
Compatible with pandas 2.2.3 and numpy 2.1.3
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_netflix_data(n_users=1000, n_movies=500, n_ratings=50000):
    """Generate synthetic movie ratings data"""
    print("Generating synthetic Netflix data...")
    
    # Generate users
    users = [f"user_{i}" for i in range(n_users)]
    
    # Generate movies with genres
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi']
    movies = []
    for i in range(n_movies):
        movie = {
            'movie_id': f"movie_{i}",
            'title': f"Movie Title {i}",
            'genre': np.random.choice(genres),
            'year': int(np.random.randint(1990, 2024))  # Convert to int for pandas 2.2.3
        }
        movies.append(movie)
    
    # Generate ratings
    ratings = []
    base_date = datetime.now()
    for _ in range(n_ratings):
        rating = {
            'user_id': np.random.choice(users),
            'movie_id': np.random.choice([m['movie_id'] for m in movies]),
            'rating': int(np.random.randint(1, 6)),  # 1-5 stars
            'timestamp': base_date - timedelta(days=int(np.random.randint(0, 365)))
        }
        ratings.append(rating)
    
    # Create DataFrames
    movies_df = pd.DataFrame(movies)
    ratings_df = pd.DataFrame(ratings)
    
    # Save to raw data folder
    os.makedirs('D:/Netflix/data/raw', exist_ok=True)
    movies_df.to_csv('D:/Netflix/data/raw/movies.csv', index=False)
    ratings_df.to_csv('D:/Netflix/data/raw/ratings.csv', index=False)
    
    print(f"Generated {len(movies)} movies and {len(ratings)} ratings")
    print(f"Files saved to D:/Netflix/data/raw/")
    return movies_df, ratings_df

if __name__ == "__main__":
    generate_netflix_data()