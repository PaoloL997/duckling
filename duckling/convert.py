"""Document processing module for PDF and image analysis.

This module provides DocumentProcessor and GenericProcessor classes for converting,
chunking, and analyzing various document types (PDFs, images, markdown, CSV, etc.)
using Docling and OpenAI APIs.
"""

import json
import re
import base64
from pathlib import Path
import tiktoken
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import ImageRefMode
from langchain.schema import Document, HumanMessage
from langchain_openai import ChatOpenAI

from .base import BaseDocumentConverter
from .utilities import setup_logger
from .config import Config

logger = setup_logger(__name__)
load_dotenv()

cfg = Config()


class DocumentProcessor(BaseDocumentConverter):
    """Processes PDF documents into chunked text and image descriptions.

    Extends BaseDocumentConverter to handle PDF conversion, text chunking,
    markdown export, and LLM-based image description extraction.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        llm_max_tokens: int = 900_000,
    ):
        """Initialize the document processor.

        Args:
            max_tokens: Maximum number of tokens per text chunk. Defaults to 4096.
            llm_max_tokens: Maximum token limit for LLM inputs. Defaults to 900000.
        """
        load_dotenv()
        super().__init__(max_tokens=max_tokens)
        self.llm_model_name = cfg.models("llm")
        self.llm_max_tokens = llm_max_tokens

        self.llm = ChatOpenAI(model=self.llm_model_name)
        self.image_prompt = cfg.prompts("retrive_image_description")

    def save_as_markdown(self, document, md_filepath: Path):
        """Save a Docling document as markdown file with embedded images.

        Args:
            document: A DoclingDocument object to save.
            md_filepath: Path where the markdown file will be saved.

        Returns:
            Path: The path to the saved markdown file.
        """
        document.save_as_markdown(
            filename=str(md_filepath),
            image_mode=ImageRefMode.REFERENCED,
            artifacts_dir=Path("artifacts"),
            include_annotations=False,
        )
        return md_filepath

    def split_markdown_for_llm(self, markdown: str) -> list:
        """Split markdown content into chunks respecting LLM token limits.

        Uses tiktoken to encode and split content while staying within the LLM's
        maximum token limit for processing.

        Args:
            markdown: The markdown content to split.

        Returns:
            list: List of markdown chunks, each within the token limit.
        """
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(markdown)
        return [
            encoding.decode(tokens[i : i + self.llm_max_tokens])
            for i in range(0, len(tokens), self.llm_max_tokens)
        ]

    def clean_json_response(self, content: str) -> str:
        """Clean JSON response by removing markdown code block markers.

        Args:
            content: Raw LLM response content.

        Returns:
            str: Cleaned JSON string without markdown markers.
        """
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        return content.strip()

    def extract_image_descriptions(self, chunks: list) -> list:
        """Extract image descriptions from markdown chunks using LLM.

        Processes each markdown chunk and invokes the LLM to identify and describe
        all images found within the content.

        Args:
            chunks: List of markdown chunks to process.

        Returns:
            list: List of dictionaries containing image metadata and descriptions.
        """
        all_images = []
        for i, chunk in enumerate(chunks, start=1):
            query = self.image_prompt.format(markdown_content=chunk)
            response = self.llm.invoke([HumanMessage(content=query)])
            try:
                content = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                cleaned = self.clean_json_response(content)
                chunk_images = json.loads(cleaned)
                if isinstance(chunk_images, list):
                    all_images.extend(chunk_images)
                else:
                    logger.warning("Chunk %d did not return a list.", i)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse JSON from chunk %d: %s", i, e)
        return all_images

    def create_image_documents(self, all_images: list, filepath: str) -> list:
        """Convert extracted image data into LangChain Document objects.

        Args:
            all_images: List of image metadata dictionaries.
            filepath: Original document filepath for path construction.

        Returns:
            list: List of LangChain Document objects with image metadata.
        """
        img_docs = []
        for img in all_images:
            original_path = img.get("path", "")
            if original_path:
                image_name = Path(original_path).name
                relative_path = (
                    Path("media") / Path(filepath).stem / "artifacts" / image_name
                ).as_posix()
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
        """Run the complete document processing pipeline.

        Performs the following steps:
        1. Convert PDF to a structured Docling document.
        2. Split into text chunks.
        3. Export markdown.
        4. Extract image descriptions using an LLM.

        Args:
            filepath: Path to the input PDF file.

        Returns:
            list: All LangChain Document objects (text chunks and image descriptions).
        """
        document = super().convert_document(filepath)
        text_docs = super().chunk_document(document)

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
            "Processing complete: %d text chunks, "
            "%d image descriptions, total %d documents.",
            len(text_docs),
            len(image_docs),
            len(all_docs),
        )
        return all_docs


class GenericProcessor(BaseDocumentConverter):
    """A generic processor to handle various file types and convert them into Document objects."""

    IMG_EXTENSION = [".png", ".jpg", ".jpeg"]
    FILE_EXTENSION = [".md", ".csv"]

    def __init__(
        self,
        max_tokens: int = 4096,
    ):
        """Initialize the GenericProcessor.

        Args:
            max_tokens: Maximum number of tokens per text chunk. Defaults to 4096.
        """
        super().__init__(max_tokens=max_tokens)
        self.llm = ChatOpenAI(model=cfg.models("llm"))

    @staticmethod
    def _file_to_base64(path: str) -> str:
        """Convert a file to a base64-encoded string.

        Args:
            path: Path to the file to encode.

        Returns:
            str: Base64-encoded file content.
        """
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _describe_image(
        self,
        base64_image: str,
        prompt: str,
    ) -> str:
        """Describe an image using the LLM.

        Args:
            base64_image: Base64-encoded image data.
            prompt: Prompt for image description.

        Returns:
            str: LLM-generated image description.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
        response = self.llm.invoke([message])
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return content

    def load_pdf(self, path: str):
        """Load document using standard Docling pipeline.

        Args:
            path: Path to the PDF file.

        Returns:
            list: List of LangChain Document chunks.
        """
        document = self.convert_document(path)
        return self.chunk_document(document)

    def text2markdown(self, path: str, outpath: Path) -> str:
        """Convert a plain text file to markdown format.

        Args:
            path: Path to the text file.
            outpath: Output directory path.

        Returns:
            str: Path to the generated markdown file.
        """
        output_path = outpath / f"{Path(path).stem}.md"
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return str(output_path)

    def load_generic(self, path: str):
        """Load any document type using Docling converter.

        Args:
            path: Path to the document file.

        Returns:
            Document: The converted document object.
        """
        converter = DocumentConverter()
        return converter.convert(source=path).document

    def image2document(self, path: str):
        """Convert an image file to a Document object.

        Args:
            path: Path to the image file.

        Returns:
            Document: Document object with image description and metadata.
        """
        response = self._describe_image(
            self._file_to_base64(path), prompt=cfg.prompts("describe_image")
        )
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
            },
        )

    def convert(self, filepath: str):
        """Convert any supported document type.

        Args:
            filepath: Path to the document file.

        Returns:
            list: List of Document objects.

        Raises:
            ValueError: If file format is not supported.
        """
        ext = Path(filepath).suffix.lower()
        if ext == ".pdf":
            processor = DocumentProcessor()
            return processor.process(filepath)
        root_path = Path("media")
        root_path.mkdir(exist_ok=True)
        source_path = root_path / Path(filepath).stem
        source_path.mkdir(exist_ok=True)
        artifacts_path = source_path / "artifacts"
        artifacts_path.mkdir(exist_ok=True)
        new_filepath = source_path / Path(filepath).name
        if ext == ".txt":
            self.copy_source_file(filepath, source_path)
            md_path = self.text2markdown(str(new_filepath), outpath=source_path)
            return self.chunk_document(self.load_generic(md_path))
        if ext in self.IMG_EXTENSION:
            self.copy_source_file(filepath, source_path)
            return [self.image2document(str(new_filepath))]
        if ext in self.FILE_EXTENSION:
            self.copy_source_file(filepath, source_path)
            document = self.load_generic(str(new_filepath))
            return self.chunk_document(document)
        raise ValueError(f"Unsupported file format: {ext}")
