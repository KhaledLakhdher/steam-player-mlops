# Reproducible serving image.
# Trains + registers the model at build time so the image is fully self-contained.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY . .

# Train + register the model INSIDE the image (writes mlflow.db + mlartifacts to /app).
RUN python -m src.train

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
