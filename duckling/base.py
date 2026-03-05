"""Base converter utilities for document loading and chunking."""

from typing import List
from pathlib import Path

from transformers import AutoTokenizer

from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


class MDTableSerializerProvider(ChunkingSerializerProvider):
    """Custom serializer provider that configures a Markdown table serializer."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),  # configuring a different table serializer
        )


class BaseConverter:
    """Base converter that handles document loading and chunking.

    This class provides utilities to load documents via a
    DocumentConverter and split them into token-aware chunks.
    """

    def __init__(self):
        """Initialize the base converter."""
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            max_tokens=4096,
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
            serializer_provider=MDTableSerializerProvider(),
        )

    def load(self, path: str) -> DoclingDocument:
        """Load a document from disk and convert it to a DoclingDocument.

        Args:
            path: Filesystem path to the source document.

        Returns:
            A `DoclingDocument` representing the converted document.
        """
        converter = DocumentConverter()
        return converter.convert(source=path).document

    def chunk(
        self, document: DoclingDocument, namespace: str = "namespace"
    ) -> List[Document]:
        """Split a DoclingDocument into a list of LangChain Documents.

        Args:
            document: The input `DoclingDocument` to chunk.
            namespace: Namespace to attach to document metadata.

        Returns:
            A list of `langchain_core.documents.Document` objects.
        """
        docs = []

        chunks = list(self.chunker.chunk(dl_doc=document))

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
            # Detect whether the chunk originates from a table
            try:
                item_labels = {
                    item.label
                    for item in chunk.meta.doc_items
                    if hasattr(item, "label")
                }
                chunk_type = "table" if "table" in item_labels else "text"
            except Exception:
                chunk_type = "text"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "path": relative_path,
                        "page_start": str(page_start),
                        "page_end": str(page_end),
                        "type": chunk_type,
                        "name": filename,
                        "namespace": namespace,
                    },
                )
            )

        return docs
