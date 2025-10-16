import os
import json
import re
import logging
import tiktoken
from typing import List
from dotenv import load_dotenv

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import ImageRefMode
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from langchain.schema import Document, HumanMessage
from langchain_openai import ChatOpenAI


# === Logging configuration ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/document_processor.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    A class for converting, chunking, and analyzing PDF documents using Docling and OpenAI.
    Extracts text and image data and prepares them as LangChain documents for further processing.
    """

    def __init__(
        self,
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 4096,
        llm_model_name: str = "gpt-4.1-mini",
        llm_max_tokens: int = 900_000
    ):
        """
        Initialize the document processor.

        Args:
            embed_model_id: Hugging Face model ID used for text embeddings.
            max_tokens: Maximum number of tokens per text chunk.
            llm_model_name: OpenAI model name used for image description.
            llm_max_tokens: Maximum token limit for LLM inputs.
        """
        load_dotenv()

        self.embed_model_id = embed_model_id
        self.max_tokens = max_tokens
        self.llm_model_name = llm_model_name
        self.llm_max_tokens = llm_max_tokens

        self.tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(embed_model_id),
            max_tokens=max_tokens,
        )

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            merge_peers=True,
        )

        self.llm = ChatOpenAI(
            model_name=llm_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.image_prompt = """
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

    def convert_document(self, source_path: str):
        """
        Convert a PDF file into a structured Docling document.

        Args:
            source_path: Path to the input PDF file.

        Returns:
            A Docling document object.
        """
        logger.info(f"Converting document: {source_path}")
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
        logger.info("Chunking document into smaller pieces...")
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
                    },
                )
            )
        logger.info(f"Created {len(docs)} text chunks.")
        return docs

    def save_as_markdown(self, document, output_path: str = "output.md") -> str:
        """
        Export a Docling document as markdown.

        Args:
            document: The Docling document object.
            output_path: Path to save the generated markdown file.

        Returns:
            The markdown content as a string.
        """
        logger.info(f"Saving document as markdown to {output_path}")
        document.save_as_markdown(
            filename=output_path,
            image_mode=ImageRefMode.REFERENCED,
            include_annotations=False,
        )
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    def split_markdown_for_llm(self, markdown: str) -> List[str]:
        """
        Split markdown content into LLM-compatible chunks based on token limits.

        Args:
            markdown: The complete markdown text.

        Returns:
            A list of markdown chunks within token limits.
        """
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(markdown)
        chunks = []
        for i in range(0, len(tokens), self.llm_max_tokens):
            chunks.append(encoding.decode(tokens[i:i + self.llm_max_tokens]))
        return chunks

    def clean_json_response(self, content: str) -> str:
        """
        Clean a raw JSON string returned by the LLM.

        Args:
            content: The raw LLM response.

        Returns:
            A cleaned JSON string ready for parsing.
        """
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        return content.strip()

    def extract_images_from_markdown(self, markdown: str) -> List[Document]:
        """
        Extract image data from markdown using an LLM.

        Args:
            markdown: Markdown text content.

        Returns:
            A list of LangChain Document objects describing the images.
        """
        logger.info("Extracting image data from markdown...")
        text_chunks = self.split_markdown_for_llm(markdown)
        all_images = []

        for i, chunk in enumerate(text_chunks, start=1):
            query = self.image_prompt.format(markdown_content=chunk)
            response = self.llm.invoke([HumanMessage(content=query)])

            try:
                cleaned = self.clean_json_response(response.content)
                chunk_images = json.loads(cleaned)
                if isinstance(chunk_images, list):
                    all_images.extend(chunk_images)
                else:
                    logger.warning(f"Chunk {i} did not return a list.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from chunk {i}: {e}")

        img_docs = [
            Document(
                page_content=img["description"],
                metadata={
                    "path": img.get("path"),
                    "pages": None,
                    "type": "image",
                    "name": img.get("name"),
                },
            )
            for img in all_images
        ]
        logger.info(f"Extracted {len(img_docs)} image descriptions.")
        return img_docs

    def process(self, source_path: str, output_md_path: str = "output.md"):
        """
        Run the complete document processing pipeline:
        1. Convert PDF to a structured Docling document.
        2. Split into text chunks.
        3. Export markdown.
        4. Extract image descriptions using an LLM.

        Args:
            source_path: Path to the input PDF file.
            output_md_path: Path to save the markdown output.
        Returns:
            A list of all LangChain Document objects (text chunks and image descriptions).
        """
        document = self.convert_document(source_path)
        text_docs = self.chunk_document(document)
        markdown = self.save_as_markdown(document, output_md_path)
        image_docs = self.extract_images_from_markdown(markdown)
        all_docs = text_docs + image_docs

        logger.info(
            f"Processing complete: {len(text_docs)} text chunks, "
            f"{len(image_docs)} image descriptions, total {len(all_docs)} documents."
        )
        return all_docs
