FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.0
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Imposta la cache HF in una directory fissa nell'immagine
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV DOCLING_ARTIFACTS_PATH=/app/models

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

COPY . .

RUN poetry install --only main --no-interaction --no-ansi

# Download modelli esistenti
RUN python -c "from duckling.graph import DucklingGraph; DucklingGraph()"
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# Pre-scarica i modelli Docling
RUN python -c "
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PipelineOptions

pipeline_options = PipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

converter = DocumentConverter()
print('Modelli Docling scaricati correttamente.')
"

# A runtime forza l'uso della cache locale, niente chiamate a HF
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]