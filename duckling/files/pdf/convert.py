"""PDF conversion utilities built on top of Docling and LLMs."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from docling.datamodel.document import DoclingDocument

from duckling.service import CloudService
from duckling.base import BaseConverter
from duckling.files.pdf.options import ImageOptions
from duckling.files.pdf.describe import DescribeImages

load_dotenv()


class PDF:
    """PDF converter that extracts text and images into documents.

    Uses docling to convert PDFs, saves markdown, filters images, and
    delegates image description to `DescribeImages` when artifacts exist.
    """

    def __init__(
        self,
        context_window: int = 900_000,
        model: str = "gpt-4.1-nano",
    ):
        self.llm = ChatOpenAI(model=model)
        self.describe = DescribeImages(model=model, max_tokens=context_window)
        self.cloud = CloudService()
        self.base = BaseConverter()

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
        source = Path("media") / Path(path).stem
        document = self.cloud.load_pdf(path)

        text_chunks = self.base.chunk(document=document, namespace=namespace)
        md_filepath = source / f"{Path(path).stem}.md"
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
                markdown=cleaned_markdown,
                source=str(source),
                path=path,
                namespace=namespace,
            )

        return text_chunks + image_chunks
