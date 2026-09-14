FROM python:3.10-slim

# System dependencies for OpenCV and Image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download VGG-Face weights taaki runtime par download na karna pade
RUN python -c "from deepface import DeepFace; DeepFace.build_model('VGG-Face')"

# Copy app code
COPY . .

# Expose port
EXPOSE 5000

# Run with Gunicorn production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]