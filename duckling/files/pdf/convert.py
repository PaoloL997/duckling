"""PDF conversion utilities built on top of Docling and LLMs."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from docling_core.types.doc.document import ImageRefMode

# Group docling imports together
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

from duckling.files.pdf.options import ImageOptions
from duckling.utils import copy_source_file, create_source
from duckling.files.pdf.describe import DescribeImages
from duckling.base import BaseConverter

load_dotenv()


class PDF(BaseConverter):
    """PDF converter that extracts text and images into documents.

    Uses docling to convert PDFs, saves markdown, filters images, and
    delegates image description to `DescribeImages` when artifacts exist.
    """

    def __init__(
        self,
        max_tokens: int = 4996,
        context_window: int = 900_000,
        tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: str = "gpt-4.1-nano",
    ):
        super().__init__(
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )
        self.llm = ChatOpenAI(model=model)
        self.describe = DescribeImages(model=model, max_tokens=context_window)

    def load(self, path: str):
        """Load a PDF using docling with PDF-specific options.

        Args:
            path: Path to the PDF file.

        Returns:
            A `DoclingDocument` produced by the converter.
        """
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True, do_formula_enrichment=True, images_scale=4
        )
        accel_opts = AcceleratorOptions(device=AcceleratorDevice.CUDA, num_threads=8)
        pipeline_options.accelerator_options = accel_opts
        pipeline_options.do_ocr = True
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return converter.convert(source=path).document

    def save_as_markdown(self, document: DoclingDocument, md_filepath: Path):
        """Save the DoclingDocument as a markdown file with referenced images.

        Args:
            document: The `DoclingDocument` to save.
            md_filepath: Destination markdown filepath.
        """
        document.save_as_markdown(
            filename=str(md_filepath),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("artifacts"),
            include_annotations=False,
        )

    @staticmethod
    def filter_images(
        document: DoclingDocument,
        content: str,
        source: Path,
        md_filepath: Path,
    ) -> str:
        """Filter and clean image references in markdown and write result.

        Args:
            document: Original `DoclingDocument` used to infer page size.
            content: Markdown content to filter.
            source: Source directory containing artifacts.
            md_filepath: Path where cleaned markdown will be written.

        Returns:
            Cleaned markdown string.
        """
        options = ImageOptions(
            page_width=int(
                list(document.pages.values())[0].size.width * 4
            ),  # Image scale == 4
            page_height=int(list(document.pages.values())[0].size.height * 4),
            min_size_ratio=0.1,
        )
        cleaned_markdown = options.filter_images(
            markdown_content=content,
            source_path=source,
        )
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write(cleaned_markdown)

        return cleaned_markdown

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a PDF into text and image-derived document chunks.

        Args:
            path: Path to the PDF file.
            namespace: Namespace to attach to produced documents.

        Returns:
            A list of `Document` objects extracted from the PDF.
        """
        source = create_source(path)
        copy_source_file(path, source)
        document = self.load(path)
        text_chunks = self.chunk(document, namespace=namespace)
        md_filepath = source / f"{Path(path).stem}.md"
        self.save_as_markdown(document, md_filepath)
        with open(md_filepath, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        cleaned_markdown = self.filter_images(
            document=document,
            content=markdown_content,
            source=source,
            md_filepath=md_filepath,
        )

        image_chunks = []
        if any((source / "artifacts").iterdir()):
            image_chunks = self.describe.run(
                markdown=cleaned_markdown, source=source, path=path, namespace=namespace
            )
        chunks = text_chunks + image_chunks
        return chunks
