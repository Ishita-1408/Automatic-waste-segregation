# ── Build stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flask pillow

# Copy source
COPY src/        ./src/
COPY models/model_meta.json ./models/model_meta.json
# Model weights should be mounted or copied separately:
# COPY models/best_model.pth ./models/best_model.pth

EXPOSE 5000

ENV FLASK_ENV=production
ENV MODEL_PATH=models/best_model.pth
ENV META_PATH=models/model_meta.json

CMD ["python", "src/app.py"]
