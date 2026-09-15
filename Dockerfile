FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download lightweight Facenet512 weights during Docker build
RUN python -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

COPY . .

EXPOSE 5000

# Single worker with 2 threads to keep RAM under 300MB
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "--threads", "2", "app:app"]