FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Official HuggingFace mirror se verified ONNX model fetch karein
RUN mkdir -p /app/models && \
    wget -O /app/models/facenet.onnx https://huggingface.co/qualcomm/FaceNet/resolve/main/facenet.onnx

COPY . .

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 180 --workers 1 app:app"]