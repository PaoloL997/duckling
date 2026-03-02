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
ENV TRANSFORMERS_CACHE=/app/models
ENV DOCLING_ARTIFACTS_PATH=/app/models
ENV EASYOCR_MODULE_PATH=/app/models/EasyOcr

WORKDIR /app

COPY pyproject.toml poetry.lock* README.md ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi --no-root

COPY . .

RUN poetry install --only main --no-interaction --no-ansi

# Download modelli duckling
RUN python -c "from duckling.graph import DucklingGraph; DucklingGraph()"

# Download modelli EasyOCR
RUN python -c "import easyocr; reader = easyocr.Reader(['it', 'en'], model_storage_directory='/app/models/EasyOcr', download_enabled=True); print('EasyOCR models scaricati.')"

# Download modelli Docling
RUN python -c "from docling.document_converter import DocumentConverter, PdfFormatOption; from docling.datamodel.pipeline_options import PdfPipelineOptions; from docling.datamodel.base_models import InputFormat; pipeline_options = PdfPipelineOptions(do_ocr=True, generate_picture_images=True, do_formula_enrichment=True); converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}); print('Modelli Docling scaricati correttamente.')"

# A runtime usa solo la cache locale
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]