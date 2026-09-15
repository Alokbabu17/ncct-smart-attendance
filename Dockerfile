FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download lightweight Facenet512 weights
RUN python -c "import os; os.environ['CUDA_VISIBLE_DEVICES']='-1'; from deepface import DeepFace; DeepFace.build_model('Facenet512')"

COPY . .

EXPOSE 5000

# Strict 1 worker, sync mode, 180s timeout taaki image load hone par gunicorn kill na kare
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "180", "--workers", "1", "--worker-class", "sync", "app:app"]