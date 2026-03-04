"""PDF conversion utilities built on top of Docling and LLMs."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from docling.datamodel.document import DoclingDocument
from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

from duckling.service import CloudService
from duckling.files.pdf.options import ImageOptions
from duckling.files.pdf.describe import DescribeImages

load_dotenv()


class MDTableSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider che usa Markdown per le tabelle."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


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

        chunker = HierarchicalChunker(serializer_provider=MDTableSerializerProvider())
        text_chunks = [
            Document(
                page_content=chunker.contextualize(raw_chunk),
                metadata={
                    "source": path,
                    "namespace": namespace,
                },
            )
            for raw_chunk in chunker.chunk(document)
            if chunker.contextualize(raw_chunk).strip()
        ]

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
