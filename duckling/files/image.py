"""Image converter: LLM-based image description utilities."""

from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from duckling.utils import copy_source_file, file_to_base64, create_source

PROMPT = """
Describe the image in detail. Explain what is visible, including the main subjects, their surroundings, and any relevant actions, 
objects, or features. Mention colors, composition, and general atmosphere if noticeable. If any measurements, dimensions, scale
indicators, abbreviations, symbols, numbers, or text appear in the image, include them accurately in the description. Focus on
providing a clear, precise, and complete account of everything observable in the image.
"""


class Image:
    """Convert images into descriptive text documents using an LLM.

    The converter encodes the image as base64, sends it to an LLM prompt,
    and returns a `Document` containing the model-generated description.
    """

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.llm = ChatOpenAI(model=model)

    @staticmethod
    def format_message(prompt: str, image_str: str) -> HumanMessage:
        """Format a HumanMessage containing text and an inline base64 image.

        Args:
            prompt: Textual prompt to send to the LLM.
            image_str: Base64-encoded image string.

        Returns:
            A `HumanMessage` ready for the LLM invocation.
        """
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_str}"},
                },
            ]
        )
        return message

    def describe(self, path: str):
        """Describe an image file by querying the LLM.

        Args:
            path: Path to the image file.

        Returns:
            The LLM-generated description as a string.
        """
        base64_image = file_to_base64(path)
        message = self.format_message(PROMPT, base64_image)
        response = self.llm.invoke([message])
        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        return content

    def convert(self, path: str, namespace: str) -> list[Document]:
        """Convert an image into a `Document` with descriptive text.

        Args:
            path: Path to the image file.
            namespace: Namespace to attach to the document metadata.

        Returns:
            A list containing one `Document` with the image description.
        """
        source = create_source(path)
        copy_source_file(path, source)
        description = self.describe(path)
        relative_path = (Path("media") / Path(path).stem / Path(path).name).as_posix()
        document = Document(
            page_content=description,
            metadata={
                "path": relative_path,
                "page_start": "N/A",
                "page_end": "N/A",
                "type": "image",
                "name": Path(path).name,
                "namespace": namespace,
            },
        )
        return [document]
