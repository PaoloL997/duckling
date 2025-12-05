import os
import json
import re
import logging
import tiktoken
import base64
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import ImageRefMode
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
    def save_as_markdown(self, document, md_filepath: Path):
        document.save_as_markdown(
            filename=str(md_filepath),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("artifacts"), # Crea una dir all'interno di media/filename
            include_annotations=False,
        )
        return md_filepath

    def split_markdown_for_llm(self, markdown: str) -> list:
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(markdown)
        return [encoding.decode(tokens[i:i + self.llm_max_tokens]) for i in range(0, len(tokens), self.llm_max_tokens)]

    def clean_json_response(self, content: str) -> str:
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*$', '', content)
        return content.strip()

    def extract_image_descriptions(self, chunks: list) -> list:
        all_images = []
        for i, chunk in enumerate(chunks, start=1):
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
        return all_images
    
    def create_image_documents(self, all_images: list, filepath: str) -> list:
        img_docs = []
        for img in all_images:
            original_path = img.get("path", "")
            if original_path:
                image_name = Path(original_path).name
                relative_path = (Path("media") / Path(filepath).stem / "artifacts" / image_name).as_posix()
            else:
                relative_path = ""
            img_docs.append(
                Document(
                    page_content=img["description"],
                    metadata={
                        "path": relative_path,
                        "page_start": "N/A",
                        "page_end": "N/A",
                        "type": "image",
                        "name": img.get("name"),
                        "namespace": "CaseDoneDemo",
                    },
                )
            )
        return img_docs

    def process(self, filepath: str):
        """
        Run the complete document processing pipeline:
        1. Convert PDF to a structured Docling document.
        2. Split into text chunks.
        3. Export markdown.
        4. Extract image descriptions using an LLM.

        Args:
            filepath: Path to the input PDF file.
        Returns:
            A list of all LangChain Document objects (text chunks and image descriptions).
        """
        document = super().convert_document(filepath)
        text_docs = super().chunk_document(document)
        # Setup paths
        root_path = Path("media")
        root_path.mkdir(exist_ok=True)
        source_path = root_path / Path(filepath).stem
        source_path.mkdir(exist_ok=True)
        artifacts_path = source_path / "artifacts"
        artifacts_path.mkdir(exist_ok=True)

        self.copy_source_file(filepath, source_path)
        md_filepath = source_path / f"{Path(filepath).stem}.md"
        self.save_as_markdown(document, md_filepath)

        with open(md_filepath, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        llm_chunks = self.split_markdown_for_llm(markdown_content)
        all_images = self.extract_image_descriptions(llm_chunks)
        image_docs = self.create_image_documents(all_images, filepath)

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
        
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    @staticmethod
    def _file_to_base64(path: str) -> str:
        """Convert a file to a base64-encoded string."""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _describe_image(
            self,
            base64_image: str,
            prompt: str,
            ) -> Document:
        # TODO: improve image prompt
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        )
        response = self.llm.invoke([message]).content
        return response

    def load_pdf(self, path: str):
        """Load document using standard Docling pipeline."""
        # Qui teoricamente già popola i chunks come vogliamo noi
        document = self.convert_document(path)
        return self.chunk_document(document)

    def text2markdown(self, path: str, outpath: Path) -> str:
        """Convert a plain text file to markdown format."""
        output_path = outpath / f"{Path(path).stem}.md"
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        output_path.write_text("\n".join(lines), encoding='utf-8')
        return str(output_path)
    
    def load_generic(self, path: str):
        converter = DocumentConverter()
        return converter.convert(source=path).document

    def image2document(self, path: str):
        prompt = """
        Describe the image in detail. Explain what is visible, including the main subjects, their surroundings, and any relevant actions, 
        objects, or features. Mention colors, composition, and general atmosphere if noticeable. If any measurements, dimensions, scale
        indicators, abbreviations, symbols, numbers, or text appear in the image, include them accurately in the description. Focus on
        providing a clear, precise, and complete account of everything observable in the image.
        """
        response = self._describe_image(self._file_to_base64(path), prompt=prompt)
        relative_path = (Path("media") / Path(path).stem / Path(path).name).as_posix()
        return Document(
            page_content=response,
            metadata={
                "path": relative_path,
                "page_start": "N/A",
                "page_end": "N/A",
                "type": "image",
                "name": Path(path).name,
                "namespace": "CaseDoneDemo",
            }
        )

    def convert(self, filepath: str):

        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            processor = DocumentProcessor()
            return processor.process(filepath)
        else:
            root_path = Path("media")
            root_path.mkdir(exist_ok=True)
            source_path = root_path / Path(filepath).stem
            source_path.mkdir(exist_ok=True)
            artifacts_path = source_path / "artifacts"
            artifacts_path.mkdir(exist_ok=True)

            new_filepath = source_path / Path(filepath).name

            if ext == ".txt":
                self.copy_source_file(filepath, source_path)
                md_path = self.text2markdown(new_filepath, outpath=source_path)
                return self.load_document(md_path)
            elif ext in self.IMG_EXTENSION:
                self.copy_source_file(filepath, source_path)
                return [self.image2document(new_filepath)]
            elif ext in self.FILE_EXTENSION:
                self.copy_source_file(filepath, source_path)
                document = self.load_generic(new_filepath)
                return self.chunk_document(document)
            else:
                raise ValueError(f"Unsupported file format: {ext}")        
        





