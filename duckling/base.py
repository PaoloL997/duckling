"""Base converter utilities for document loading and chunking."""

from typing import List
from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import (
    HierarchicalChunker,
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc import DoclingDocument

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


class MDTableSerializerProvider(ChunkingSerializerProvider):
    """Serializer provider che usa Markdown per le tabelle."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class BaseConverter:
    """Base converter that handles document loading and chunking.

    This class provides utilities to load documents via a
    DocumentConverter and split them into token-aware chunks.
    """

    def __init__(self):
        """Initialize the base converter."""
        self.chunker = HierarchicalChunker(
            serializer_provider=MDTableSerializerProvider()
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

        # Handle tables if document has no text content
        if len(document.texts) == 0 and len(document.tables) > 0:
            # For table-only documents, chunk tables directly
            for table_idx, table in enumerate(document.tables):
                # Convert table to markdown for better readability
                table_content = table.export_to_markdown(doc=document)

                filename = document.name or "table"
                relative_path = str(
                    (Path("media") / filename / f"table_{table_idx}.md").as_posix()
                )

                docs.append(
                    Document(
                        page_content=table_content,
                        metadata={
                            "path": relative_path,
                            "page_start": "N/A",
                            "page_end": "N/A",
                            "type": "table",
                            "name": f"table_{table_idx}",
                            "namespace": namespace,
                        },
                    )
                )
        else:
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
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "path": relative_path,
                            "page_start": str(page_start),
                            "page_end": str(page_end),
                            "type": "text",
                            "name": filename,
                            "namespace": namespace,
                        },
                    )
                )

        return docs
