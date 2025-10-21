import os

from dotenv import load_dotenv
from typing import List

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain.schema import Document


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


IMAGE_PROMPT = """
You are given the following markdown content extracted from a PDF document. 
Your task is to identify all images present in the markdown.

For each image, create a JSON object with the following fields:
- "path": the full path or URL of the image as it appears in the markdown
- "name": the name or identifier used to reference this image in the text (e.g., "Figure 1", "Diagram A", etc.)
- "description": a detailed, reasoned explanation of what the image represents based on its surrounding context and purpose.

IMPORTANT: If the same image appears multiple times, create only one entry with a merged, comprehensive description.

Output must be a single valid JSON array.
Markdown content:
{markdown_content}
"""

class BaseDocumentConverter:
    def __init__(
            self,
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            max_tokens: int = 4096
    ):
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        
        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embedding_model),
            max_tokens=max_tokens
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
        )

    def convert_document(self, source_path: str):
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            do_formula_enrichment=True,
            image_scale=2
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        return converter.convert(source=source_path).document
    

    def chunk_document(self, document) -> List[Document]:
        """
        Split a Docling document into smaller text chunks.

        Args:
            document: A Docling document object.

        Returns:
            A list of LangChain Document objects containing text chunks.
        """
        chunks = list(self.chunker.chunk(dl_doc=document))
        docs = []

        for chunk in chunks:
            content = self.chunker.contextualize(chunk=chunk)
            filepath = chunk.meta.origin.filename
            filename = os.path.splitext(filepath)[0]
            page_start = chunk.meta.doc_items[0].prov[0].page_no
            page_end = chunk.meta.doc_items[-1].prov[-1].page_no

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "path": filepath,
                        "pages": [page_start, page_end],
                        "type": "text",
                        "name": filename,
                        "namespace": "CaseDoneDemo" # Devi usare lo stesso quando crei una instance di MilvusStore (agent_rag)
                    },
                )
            )
        return docs