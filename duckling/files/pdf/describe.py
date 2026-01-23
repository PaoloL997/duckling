"""Extract and refine image descriptions from PDF-generated markdown."""

import re
import json
from pathlib import Path

import tiktoken
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from duckling.utils import file_to_base64

DESCRIPTION_PROMPT = """
You are given the following markdown content extracted from a PDF document.

    Your task is to identify and process images ONLY when they are explicitly defined using the following Markdown syntax:

    ![Image](path/to/image)

    IMPORTANT RULES:
    - Consider an image present ONLY if the exact Markdown pattern `![Image](...)` exists.
    - Ignore any textual references to figures, diagrams, screenshots, or images that are NOT accompanied by this syntax.
    - Do NOT infer images from captions, figure numbers, or descriptive text alone.

    For each valid image found, create a JSON object with the following fields:
    - "path": the full path or URL of the image exactly as it appears inside the parentheses
    - "name": the exact name or identifier used to reference this image in the surrounding text (before or after the image).  
      - If no explicit name or identifier is present, use the image filename (without extension) as the name.
    - "description": a detailed and reasoned explanation of what the image represents, inferred strictly from the surrounding text and its role in the document.

    Additional constraints:
    - Carefully analyze the text immediately before and after each image to determine its correct reference name and purpose.
    - If the same image path appears multiple times, output only one JSON object with a merged and comprehensive description.
    - Do not hallucinate or invent images that are not explicitly present via the Markdown syntax.

    Output format:
    - Return a single valid JSON array.
    - Do not include any additional text outside the JSON.

    Markdown content:
    {markdown_content}
"""

REFINE_PROMPT = """
The attached image corresponds to the following description: {description}.

    Carefully analyze the image and expand the provided description by including all visible details shown in the image.
    Make sure to cover text, symbols, labels, measurements, numerical values, annotations, and any other relevant visual elements present in the image.
    The resulting description should be comprehensive, precise, and faithful to what is visually observable.

    Return the enhanced description only, without any additional commentary or formatting.
"""


class DescribeImages:
    """Extract and refine image descriptions from markdown content.

    This class identifies image markdown patterns, requests an LLM to
    extract initial descriptions, then refines them using the image
    artifacts when available.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        max_tokens: int = 900_000,
    ):
        self.llm = ChatOpenAI(model=model)
        self.llm_max_tokens = max_tokens

    def split_markdown(self, markdown: str):
        """Split large markdown into chunks that fit the LLM token limit.

        Args:
            markdown: Full markdown content to split.

        Returns:
            List of markdown substrings within the token limit.
        """
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(markdown)
        return [
            encoding.decode(tokens[i : i + self.llm_max_tokens])
            for i in range(0, len(tokens), self.llm_max_tokens)
        ]

    def clean_json_response(self, content: str):
        """Strip markdown code fences from a JSON response string.

        Args:
            content: Raw LLM response that may include code fences.

        Returns:
            Clean JSON string suitable for parsing.
        """
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)
        return content.strip()

    def extract_descriptions(self, chunks: list):
        """Query the LLM to extract image descriptions from markdown chunks.

        Args:
            chunks: List of markdown chunks to analyze.

        Returns:
            List of image description dicts extracted from the chunks.
        """
        descriptions = []
        for chunk in chunks:
            query = DESCRIPTION_PROMPT.format(markdown_content=chunk)
            response = self.llm.invoke([HumanMessage(content=query)])
            cleaned = self.clean_json_response(str(response.content))
            chunk_images = json.loads(cleaned)
            if isinstance(chunk_images, list):
                descriptions.extend(chunk_images)
        return descriptions

    def refine_descriptions(self, images: list, source: str):
        """Refine extracted descriptions using the actual image artifacts.

        Args:
            images: List of image metadata dicts with initial descriptions.
            source: Source directory where image artifacts are stored.

        Returns:
            List of refined description dicts containing `path`, `description`, and `name`.
        """
        descriptions = []
        for img in images:
            path = img.get("path", "")
            fullpath = Path(source) / path
            desc_text = img.get("description", "")
            if not fullpath.exists():
                continue

            base64_image = file_to_base64(fullpath)
            query = REFINE_PROMPT.format(description=desc_text)
            messages = HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
            response = self.llm.invoke([messages]).content
            descriptions.append(
                {
                    "path": path,
                    "description": response,
                    "name": img.get("name", ""),
                }
            )
        return descriptions

    def create_documents(self, images: list, path: str, namespace: str = "namespace"):
        """Create `Document` objects from refined image descriptions.

        Args:
            images: List of image dicts with `description` and `path`.
            path: Original PDF path used to compute relative media paths.
            namespace: Namespace to attach to document metadata.

        Returns:
            List of `Document` objects representing each image description.
        """
        img_docs = []
        for img in images:
            original_path = img.get("path", "")
            if original_path:
                image_name = Path(original_path).name
                relative_path = (
                    Path("media") / Path(path).stem / "artifacts" / image_name
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
                        "namespace": namespace,
                    },
                )
            )
        return img_docs

    def run(self, markdown: str, source: str, path: str, namespace: str = "namespace"):
        """End-to-end extraction and refinement of image descriptions.

        Args:
            markdown: Markdown content extracted from the PDF.
            source: Directory where image artifacts are stored.
            path: Original PDF path for naming.
            namespace: Namespace to attach to created documents.

        Returns:
            List of `Document` objects describing images found in the markdown.
        """
        chunks = self.split_markdown(markdown)
        descriptions = self.extract_descriptions(chunks)
        refined_descriptions = self.refine_descriptions(
            images=descriptions,
            source=source,
        )
        image_chunks = self.create_documents(
            images=refined_descriptions, path=path, namespace=namespace
        )
        return image_chunks
