FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY config ./config
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[server]" \
    && pip install torch --index-url https://download.pytorch.org/whl/cu121
    
ENV PYTHONPATH=/app/src
ENV UMS_CONFIG=/app/config/app.example.yaml

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
