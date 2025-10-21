import os
import json
import re
import logging
import tiktoken
import base64
from typing import List
from pathlib import Path
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

from .base import BaseDocumentConverter


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


class DocumentProcessor(BaseDocumentConverter):
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
        
        # Initialize the base class
        super().__init__(embedding_model=embed_model_id, max_tokens=max_tokens)

        self.llm_model_name = llm_model_name
        self.llm_max_tokens = llm_max_tokens

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
                    "page_start": "N/A",
                    "page_end": "N/A",
                    "type": "image",
                    "name": img.get("name"),
                    "namespace": "CaseDoneDemo", # Devi usare lo stesso quando crei una instance di MilvusStore (agent_rag)
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
        document = super().convert_document(source_path)
        text_docs = super().chunk_document(document)
        markdown = self.save_as_markdown(document, output_md_path)
        image_docs = self.extract_images_from_markdown(markdown)
        all_docs = text_docs + image_docs

        logger.info(
            f"Processing complete: {len(text_docs)} text chunks, "
            f"{len(image_docs)} image descriptions, total {len(all_docs)} documents."
        )
        return all_docs
    

class GenericProcessor(BaseDocumentConverter):
    """A generic processor to handle various file types and convert them into Document objects."""

    IMG_EXTENSION = [".png", ".jpg", ".jpeg"]
    FILE_EXTENSION = [".md", ".csv"]

    def __init__(self, 
                 llm_model: str = "gpt-4.1-mini",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 4096):
        super().__init__(embedding_model=embedding_model, max_tokens=max_tokens)
        
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.llm = ChatOpenAI(model=llm_model, openai_api_key=self.OPENAI_API_KEY)

    @staticmethod
    def _file_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _image_to_document(self, base64_image: str, prompt="Describe the following image.", source=None) -> Document:
        from langchain.schema import HumanMessage
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        )
        response = self.llm.invoke([message]).content
        return Document(page_content=response, metadata={"source": source})

    def load_document(self, path: str):
        """Load document using standard Docling pipeline."""
        document = self.convert_document(path)
        return self.chunk_document(document)

    def text2markdown(self, path: str, output_dir: str = "outputs") -> str:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        md_path = output_dir / Path(path).with_suffix('.md').name
        md_path.write_text("\n".join(lines), encoding='utf-8')

        return str(md_path)


    def image2document(self, path: str, context: str = None):
        prompt = f"Describe the image knowing that it is connected to the following context: {context}" if context else "Describe the following image."
        return self._image_to_document(self._file_to_base64(path), prompt=prompt, source=path)

    def pdf2documents(self, path: str, output_dir: str = "outputs", zoom: int = 2):
        import fitz
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf = fitz.open(path)
        docs = []

        for i, page in enumerate(pdf, start=1):
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_base64 = base64.b64encode(pix.tobytes("png")).decode('utf-8')
            doc = self._image_to_document(img_base64, prompt="Convert this PDF page to Markdown.", source=path)
            doc.page_content = f"# Page {i}\n\n{doc.page_content}"
            doc.metadata["page"] = i
            docs.append(doc)

        pdf.close()
        return docs

    def convert(self, path: str, output_dir: str = "outputs"):
        ext = Path(path).suffix.lower()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if ext == ".txt":
            md_path = self.text2markdown(path, output_dir=str(output_dir))
            return self.load_document(md_path)
        elif ext in self.FILE_EXTENSION:
            return self.load_document(path)
        elif ext in self.IMG_EXTENSION:
            return [self.image2document(path)]
        elif ext == ".pdf":
            return self.pdf2documents(path, output_dir=str(output_dir))
        else:
            raise ValueError(f"Unsupported file format: {ext}")        
        





