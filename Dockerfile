FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Cache directories
ENV HF_HOME=/app/models
ENV DOCLING_ARTIFACTS_PATH=/app/models
ENV EASYOCR_MODULE_PATH=/app/models/EasyOcr

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

COPY . .

RUN poetry install --only main --no-interaction --no-ansi

# Crea la cartella per i modelli
RUN mkdir -p /app/models

# Download tutti i modelli (duckling, EasyOCR, Docling)
RUN python warmup.py

# A runtime usa solo la cache locale
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]