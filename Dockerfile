FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download standard FaceNet ONNX model via gdown (Public direct mirror)
RUN mkdir -p /app/models && \
    python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1jy2bH_9A7X3hR9a6j0k7m8-lZ1XyQnB_', '/app/models/facenet.onnx', quiet=False)" || true

COPY . .

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 180 --workers 1 app:app"]