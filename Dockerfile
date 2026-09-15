FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download pre-converted ultra-lightweight FaceNet ONNX model (only 88MB)
RUN mkdir -p /app/models && \
    wget -O /app/models/facenet.onnx https://github.com/nknytk/face-recognition-onnx/raw/main/models/facenet.onnx

COPY . .

# Support both Render dynamic $PORT and default 5000
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 180 --workers 1 app:app"]