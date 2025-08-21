# Multi-stage build for optimized image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements-api.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create app directory
WORKDIR /app

# Copy application code
COPY src/inference/api.py .

# Copy model and data files
COPY models/knn_model.pkl ./models/
COPY data/processed/user_item_matrix.csv ./data/processed/
COPY data/processed/user_encoder.pkl ./data/processed/
COPY data/processed/movie_encoder.pkl ./data/processed/

# Create logs directory
RUN mkdir -p /app/logs

# Set environment variables
ENV MODEL_PATH=/app/models/knn_model.pkl
ENV DATA_PATH=/app/data/processed
ENV LOGS_PATH=/app/logs

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]