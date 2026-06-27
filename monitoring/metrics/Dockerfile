FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ml_metrics_exporter.py .
EXPOSE 8000
CMD ["python", "ml_metrics_exporter.py"]
