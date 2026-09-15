FROM python:3.10-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Environment flags for CPU stability
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TF_ENABLE_ONEDNN_OPTS="0"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV TF_USE_LEGACY_KERAS="1"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download lightweight Facenet512 weights
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--workers", "1", "--worker-class", "sync", "app:app"]