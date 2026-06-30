# MiDRR API — Docker container
#
# Build:  docker build -t midrr-api .
# Run:    docker run -p 8000:8000 midrr-api
#
# The container expects a trained model at models/midrr_rf.pkl.
# Mount it at runtime if you want to swap models without rebuilding:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models midrr-api

FROM python:3.11-slim

WORKDIR /app

# Install the midrr_classifier package first (its dependencies are heavier).
# Copying pyproject.toml + src before the rest of the code means Docker
# caches this layer and only re-runs pip when the package itself changes.
COPY pyproject.toml ./
COPY src/ src/

# Install package in non-editable mode (no -e flag in production)
RUN pip install --no-cache-dir .

# Install API-specific dependencies
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the code
COPY api/ api/

# Copy the trained model artifact (if it exists at build time).
# If you prefer to mount it at runtime, remove this line.
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
