"""Base document converter for PDF processing.

This module provides the BaseDocumentConverter class which handles PDF conversion,
tokenization, chunking, and preparation of documents for downstream processing.
"""

import shutil

from typing import List
from pathlib import Path

from transformers import AutoTokenizer
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from langchain_core.documents import Document

from dotenv import load_dotenv

from duckling.config import Config


load_dotenv()

cfg = Config()


class BaseDocumentConverter:
    """Base class for converting and chunking documents.

    Handles PDF conversion, tokenization using HuggingFace models, and splitting
    documents into chunks using the HybridChunker strategy.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        namespace: str = "namespace",
        config: Config | None = None,
    ):
        """Initialize the BaseDocumentConverter.

        Args:
            max_tokens: Maximum number of tokens per text chunk. Defaults to 4096.
            namespace: Namespace identifier for documents. Defaults to "namespace".
        """
        self.config = config if config else cfg
        self.max_tokens = max_tokens
        self.namespace = namespace

        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(self.config.models("tokenizer")),
            max_tokens=max_tokens,
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
        )

    def convert_document(self, source_path: str):
        """Convert a PDF document using Docling.

        Args:
            source_path: Path to the PDF file to convert.

        Returns:
            DoclingDocument: The converted document object.
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
        return converter.convert(source=source_path).document

    def copy_source_file(self, filepath: str, source_path: Path):
        """Copy source file to destination path.

        Args:
            filepath: Path to the source file.
            source_path: Destination directory path.
        """
        shutil.copy2(filepath, source_path / Path(filepath).name)

    def chunk_document(self, document: DoclingDocument) -> List[Document]:
        """Split a Docling document into LangChain Document chunks.

        Args:
            document: A DoclingDocument object to chunk.

        Returns:
            List[Document]: A list of LangChain Document objects with metadata.
        """
        chunks = list(self.chunker.chunk(dl_doc=document))
        docs = []

        for chunk in chunks:
            content = self.chunker.contextualize(chunk=chunk)
            filepath = (
                chunk.meta.origin.filename
                if hasattr(chunk.meta, "origin")
                else "unknown"
            )
            filename = Path(filepath).name
            try:
                page_start = (
                    chunk.meta.doc_items[0].prov[0].page_no
                    if hasattr(chunk.meta, "doc_items")
                    else "N/A"
                )
                page_end = (
                    chunk.meta.doc_items[-1].prov[-1].page_no
                    if hasattr(chunk.meta, "doc_items")
                    else "N/A"
                )
            except Exception:
                page_start = "N/A"
                page_end = "N/A"

            relative_path = str(
                (Path("media") / Path(filename).stem / filename).as_posix()
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "path": relative_path,
                        "page_start": str(page_start),
                        "page_end": str(page_end),
                        "type": "text",
                        "name": filename,
                        "namespace": self.namespace,
                    },
                )
            )
        return docs
