from pathlib import Path
import easyocr
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

EASYOCR_PATH = "/app/models/EasyOcr"
ARTIFACTS_PATH = Path("/app/models")


def warmup():
    # Pre-download EasyOCR models so Docling finds them (Docling sets download_enabled=False)
    print("Downloading EasyOCR models...")
    Path(EASYOCR_PATH).mkdir(parents=True, exist_ok=True)
    easyocr.Reader(["en"], model_storage_directory=EASYOCR_PATH, download_enabled=True)
    print("EasyOCR models downloaded.")

    pipeline_options = PdfPipelineOptions(
        artifacts_path=ARTIFACTS_PATH,
        generate_picture_images=True,
        do_formula_enrichment=True,
        images_scale=4,
    )
    pipeline_options.do_ocr = True
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print(converter.convert(source="2408_09869v5.pdf").document)


if __name__ == "__main__":
    warmup()
