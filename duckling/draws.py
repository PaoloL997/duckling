import statistics
import base64
from pathlib import Path
import fitz
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


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

    IMAGE_ANALYSIS_PROMPT = """
        Role: You are an expert Technical Document Analyst specialized in engineering blueprints, CAD exports, and architectural plans.
        Task: Provide a comprehensive, structured description of the attached technical drawing (File Name: {filename}). This description will serve as a searchable index for a RAG (Retrieval-Augmented Generation) system.
        Instructions: Analyze the image meticulously and provide a report covering the following sections:
        - Header & Title Block: Extract the document title, part/drawing number, author, date, and revision history.
        - Primary Subject & Scope: Describe the main assembly, component, or architectural area shown. Identify the type of drawing (e.g., assembly, schematic, P&ID, layout).
        - Tabular Data & Schedules (CRITICAL): Identify and transcribe all tables present in the drawing (e.g., Bill of Materials (BOM), Parts List, Revision Tables, or Specification Schedules). For each table, describe its columns and highlight key rows or materials listed.
        - Components & Annotations: List all labeled parts, callouts, and technical notes. Include specific references to dimensions, tolerances, and material callouts.
        - Visual Context: Describe the views provided (e.g., isometric, sectional views, details) and the spatial relationship between components.
        - Key Search Terms: Provide a list of 10-15 technical keywords found within the drawing that an expert would use to find this specific document.
        Output Style: Use precise, technical terminology. When describing tables, represent the data clearly so it can be easily indexed for text-based retrieval.
    """

    ORGANIZE_PROMPT = """
        Role: You are a Senior Technical Documentation Architect.
        Task: You have been provided with multiple individual descriptions of pages belonging to a single technical document (File Name: {filename}). Your goal is to synthesize these descriptions into a single, unified Master Summary that provides a clear overview of the entire project or assembly.
        Input Data: {list_of_page_descriptions}
        Instructions for Synthesis:
        - Global Overview: Start with a high-level summary of what the entire document represents (e.g., "This is a 5-page structural set for a bridge assembly").
        - Logical Flow: Organize the information following the document's logical progression (e.g., from General Layout to specific Component Details or BOM tables)
        - Consolidated Tables & Lists: Merge information from tables found across different pages. If Page 1 has a partial BOM and Page 5 continues it, present a unified summary of the materials and components involved.
        - Cross-Page Relationships: Explain how the pages relate to each other (e.g., "Page 2 provides a sectional view of the assembly shown in the isometric view on Page 1").
        - Technical Consistency: Ensure that part numbers, dimensions, and specifications are cross-referenced accurately. Eliminate redundant descriptions while keeping all unique technical data.
        Final Output Objective: Create a "Searchable Knowledge Base Entry" that allows a RAG system to understand the full scope of the document without reading individual page reports. Focus on technical keywords and functional relationships.
    """

    def __init__(self, model: str):
        """Initialize the DrawConverter with a specified LLM model.

        Args:
            model: The name of the OpenAI model to use.
        """
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
                        "text": self.IMAGE_ANALYSIS_PROMPT.format(filename=filename),
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
            organize_query = self.ORGANIZE_PROMPT.format(
                filename=filename, list_of_page_descriptions=pages_description
            )
            final_description = self.llm.invoke(organize_query).content
        docs = self.get_document(
            filepath=filepath, description=str(final_description), namespace=namespace
        )
        return docs
