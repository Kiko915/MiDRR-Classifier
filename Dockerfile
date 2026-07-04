# MiDRR API — Docker container
#
# Build:  docker build -t midrr-api .
# Run:    docker run -p 8000:8000 midrr-api
#
# The image bakes in a SYNTHETIC-trained placeholder model at build time
# (scripts/bootstrap_synthetic_model.py) — models/*.pkl is gitignored on
# purpose (it's a build artifact), and real Necookie telemetry isn't in yet
# (Phase 3 is still blocked). Once a real model exists, mount it over the
# baked-in one instead of rebuilding:
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models midrr-api

FROM python:3.11-slim

WORKDIR /app

# Install the midrr_classifier package first (its dependencies are heavier).
# Copying pyproject.toml + README (poetry-core's build backend reads the
# `readme` field declared in pyproject.toml and fails without it) + src
# before the rest of the code means Docker caches this layer and only
# re-runs pip when the package itself changes.
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package in non-editable mode (no -e flag in production)
RUN pip install --no-cache-dir .

# Install API-specific dependencies
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy the rest of the code
COPY api/ api/
COPY scripts/ scripts/

# Bake in a synthetic-trained placeholder model (see header comment above).
RUN python scripts/bootstrap_synthetic_model.py

# Drop root for the actual serving process.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
