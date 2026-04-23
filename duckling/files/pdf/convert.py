"""PDF conversion utilities built on top of Docling and LLMs."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from docling.datamodel.document import DoclingDocument

from duckling.service import LocalService
from duckling.base import BaseConverter
from duckling.files.pdf.options import ImageOptions
from duckling.files.pdf.describe import DescribeImages
from duckling.files.pdf.split import split_pdf, DEFAULT_PAGES_PER_CHUNK
from duckling.utils import copy_source_file

load_dotenv()

logger = logging.getLogger(__name__)


class PDF:
    """PDF converter that extracts text and images into documents.

    Large PDFs are split into smaller chunks (by top-level TOC when
    available, otherwise by a fixed page count) and processed one after
    the other, to keep memory usage bounded on the docling-serve side.
    """

    def __init__(
        self,
        context_window: int = 900_000,
        model: str = "gpt-4.1-nano",
        pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
    ):
        self.llm = ChatOpenAI(model=model)
        self.describe = DescribeImages(model=model, max_tokens=context_window)
        self.service = LocalService()
        self.base = BaseConverter()
        self.pages_per_chunk = pages_per_chunk

    @staticmethod
    def filter_images(
        document: DoclingDocument,
        content: str,
        source: Path,
        md_filepath: Path,
    ) -> str:
        """Filter and clean image references in markdown and write result."""
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

    @staticmethod
    def _shift_page_number(value, offset: int) -> str:
        """Add ``offset`` to a page-number metadata value (1-based)."""
        try:
            return str(int(value) + offset)
        except (TypeError, ValueError):
            return value

    def _rebase_text_documents(
        self,
        documents: List[Document],
        original_filename: str,
        original_stem: str,
        page_offset: int,
    ) -> None:
        """Rewrite chunk-local metadata on text documents in place."""
        relative_path = (Path("media") / original_stem / original_filename).as_posix()
        for doc in documents:
            doc.metadata["path"] = relative_path
            doc.metadata["name"] = original_filename
            doc.metadata["page_start"] = self._shift_page_number(
                doc.metadata.get("page_start"), page_offset
            )
            doc.metadata["page_end"] = self._shift_page_number(
                doc.metadata.get("page_end"), page_offset
            )

    @staticmethod
    def _merge_image_artifacts(
        documents: List[Document],
        chunk_source: Path,
        original_stem: str,
        chunk_idx: int,
    ) -> None:
        """Move chunk artifacts into the main media folder and fix metadata."""
        target_artifacts = Path("media") / original_stem / "artifacts"
        target_artifacts.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            old_rel = doc.metadata.get("path", "")
            if not old_rel:
                continue
            old_name = Path(old_rel).name
            new_name = f"chunk_{chunk_idx:04d}__{old_name}"
            new_rel = (target_artifacts / new_name).as_posix()

            src_file = chunk_source / "artifacts" / old_name
            dst_file = target_artifacts / new_name
            if src_file.exists() and not dst_file.exists():
                shutil.move(str(src_file), str(dst_file))

            doc.metadata["path"] = new_rel

    def _convert_chunk(
        self,
        chunk_path: str,
        page_offset: int,
        chunk_idx: int,
        original_path: str,
        namespace: str,
    ) -> tuple[list[Document], str]:
        """Run the full conversion pipeline on a single chunk PDF."""
        chunk_stem = Path(chunk_path).stem
        chunk_source = Path("media") / chunk_stem

        document = self.service.load_pdf(chunk_path)

        text_chunks = self.base.chunk(document=document, namespace=namespace)
        md_filepath = chunk_source / f"{chunk_stem}.md"
        with open(md_filepath, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        cleaned_markdown = self.filter_images(
            document=document,
            content=markdown_content,
            source=chunk_source,
            md_filepath=md_filepath,
        )

        image_chunks: list[Document] = []
        artifacts_dir = chunk_source / "artifacts"
        if artifacts_dir.is_dir() and any(artifacts_dir.iterdir()):
            image_chunks = self.describe.run(
                markdown=cleaned_markdown,
                source=str(chunk_source),
                path=chunk_path,
                namespace=namespace,
            )

        original_stem = Path(original_path).stem
        original_filename = Path(original_path).name

        self._rebase_text_documents(
            documents=text_chunks,
            original_filename=original_filename,
            original_stem=original_stem,
            page_offset=page_offset,
        )
        self._merge_image_artifacts(
            documents=image_chunks,
            chunk_source=chunk_source,
            original_stem=original_stem,
            chunk_idx=chunk_idx,
        )
        for doc in image_chunks:
            doc.metadata["page_start"] = self._shift_page_number(
                doc.metadata.get("page_start"), page_offset
            )
            doc.metadata["page_end"] = self._shift_page_number(
                doc.metadata.get("page_end"), page_offset
            )

        shutil.rmtree(chunk_source, ignore_errors=True)

        return text_chunks + image_chunks, cleaned_markdown

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a PDF into text and image-derived document chunks.

        The input PDF is first split into smaller chunks (by top-level
        TOC when available, otherwise every ``pages_per_chunk`` pages)
        and each chunk is processed sequentially. Chunk results are then
        merged into a single output directory under ``media/<stem>/``.
        """
        original_stem = Path(path).stem
        source = Path("media") / original_stem
        source.mkdir(parents=True, exist_ok=True)
        (source / "artifacts").mkdir(exist_ok=True)
        copy_source_file(path, source)

        all_documents: list[Document] = []
        combined_markdown_parts: list[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = split_pdf(
                path=path,
                output_dir=tmpdir,
                pages_per_chunk=self.pages_per_chunk,
            )
            logger.info("PDF split into %d chunk(s)", len(chunks))

            for chunk_idx, (chunk_path, page_offset) in enumerate(chunks, start=1):
                logger.info(
                    "Processing chunk %d/%d (page offset=%d) → %s",
                    chunk_idx,
                    len(chunks),
                    page_offset,
                    Path(chunk_path).name,
                )
                chunk_docs, chunk_md = self._convert_chunk(
                    chunk_path=chunk_path,
                    page_offset=page_offset,
                    chunk_idx=chunk_idx,
                    original_path=path,
                    namespace=namespace,
                )
                all_documents.extend(chunk_docs)
                combined_markdown_parts.append(chunk_md)

        md_filepath = source / f"{original_stem}.md"
        with open(md_filepath, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_markdown_parts))

        return all_documents
