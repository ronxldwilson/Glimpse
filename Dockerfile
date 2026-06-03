FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY object_detection/ object_detection/

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache-dir -e ".[dev]"

RUN mkdir -p models && \
    curl -sL https://github.com/ultralytics/assets/releases/download/v8.4.0/FastSAM-s.pt \
    -o models/FastSAM-s.pt

RUN python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)"

COPY encode_vocab.py .

VOLUME ["/app/models", "/data"]

ENTRYPOINT ["python", "-m"]
CMD ["object_detection.demo_discover"]
