import statistics
import base64
from pathlib import Path
import fitz
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from .config import Config


def compute_vector_density(path: str):
    """Compute vector density for a given PDF file and determine if it's a draw or text document.

    Args:
        path: Path to the PDF file.

    Returns:
        True if the document is a drawing, False if it's text.
    """
    doc = fitz.open(path)
    page_scores = []

    for page in doc:
        paths = page.get_drawings()
        words = page.get_text("words")

        p_count = len(paths)
        w_count = len(words)

        # Use a multiplier (e.g., 2x or 5x) to ensure
        # that vectors significantly outnumber text.
        # Also add a minimum path count check (e.g., > 50)
        # to ignore nearly empty pages.
        is_drawing_page = p_count > (w_count * 2)
        page_scores.append(is_drawing_page)

    if not page_scores:
        return False

    return statistics.mode(page_scores)


class DrawConverter:
    """Generate structured descriptions for technical drawings in PDF format."""

    def __init__(self, config: Config):
        """Initialize the DrawConverter with a specified LLM model.

        Args:
            model: The name of the OpenAI model to use.
        """
        self.config = config
        model = self.config.models("draw_llm")
        self.llm = ChatOpenAI(model=model)

    @staticmethod
    def page2base64(page: fitz.Page) -> str:
        """Convert a PDF page to a base64-encoded PNG image.
        Args:
            page: A fitz.Page object representing a page in the PDF.

        Returns:
            A base64-encoded string of the PNG image.
        """
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_b64

    def format_message(self, filename: str, img_b64: str) -> list[dict]:
        """Format the message for the LLM with the image and prompt.
        Args:
            filename: The name of the file being processed.
            img_b64: The base64-encoded image string.
        Returns:
            A list of message dictionaries for the LLM.
        """
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.config.prompts("draw_analysis").format(
                            filename=filename
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    },
                ],
            }
        ]
        return message

    @staticmethod
    def get_document(filepath: str, description: str, namespace: str) -> list[Document]:
        """Create a Document object with metadata.
        Args:
            filepath: The path to the original file.
            description: The generated description of the drawing.
            namespace: The namespace for the document.
        Returns:
            A list containing a single Document object.
        """
        doc = Document(
            page_content=description,
            metadata={
                "path": (
                    Path("media") / Path(filepath).stem / Path(filepath).name
                ).as_posix(),
                "page_start": "N/A",
                "page_end": "N/A",
                "type": "draw",
                "name": Path(filepath).name,
                "namespace": namespace,
            },
        )
        return [doc]

    def process(self, filepath: str, namespace: str):
        """Process the PDF file and generate structured descriptions.
        Args:
            filepath: The path to the PDF file.
            namespace: The namespace for the document.
        Returns:
            A list of Document objects.
        """
        filename = Path(filepath).stem
        doc = fitz.open(filepath)
        pages_description = []
        for page in doc:
            img_b64 = self.page2base64(page)
            message = self.format_message(filename, img_b64)
            response = self.llm.invoke(message)
            pages_description.append(response.content)

        if len(pages_description) == 1:
            final_description = pages_description[0]
        else:
            organize_query = self.config.prompts("organize_draw_description").format(
                filename=filename, list_of_page_descriptions=pages_description
            )
            final_description = self.llm.invoke(organize_query).content
        docs = self.get_document(
            filepath=filepath, description=str(final_description), namespace=namespace
        )
        return docs
