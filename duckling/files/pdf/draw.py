"""Drawing PDF converter that produces structured technical descriptions."""

from pathlib import Path
from typing import Any
import base64
import fitz
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from duckling.utils import create_source, copy_source_file

PROMPT = """
Role: You are an expert Technical Document Analyst specialized in engineering blueprints, CAD exports, and architectural plans.
    Task: Provide a high-fidelity, comprehensive, and structured description of the attached technical drawing (File Name: {filename}). This description is specifically designed to serve as a "textual twin" for a RAG (Retrieval-Augmented Generation) system.
    Core Directive (CRITICAL): Your primary goal is exhaustive identification. Do not spend effort extracting precise measurements or dimensions unless they are part of a part name or a critical specification. Instead, focus on identifying every single individual component, sub-assembly, and technical feature present. The more detail you provide about "what" is in the drawing, the higher the probability the RAG system will correctly retrieve this document for a specific query.
    Instructions: Analyze the image meticulously and provide a report covering the following sections:
    Header & Title Block:
      Extract the document title, part/drawing number, author, date, and revision history.
      Identify the organization or project name.
    Primary Subject & Scope:
      Define the main assembly or architectural area shown.
      Identify the drawing type (e.g., exploded view, assembly, schematic, P&ID, layout, section).
      Describe the overall function or purpose of the system depicted.
    Tabular Data & Schedules (MAXIMUM DETAIL):
      Transcribe all tables (Bill of Materials (BOM), Parts List, Revision Tables, Specification Schedules).
      For each table, list every row and column.
      Crucial: Ensure every part name, material, and reference code in the tables is written out clearly to enable text-based search.
    Component & Feature Inventory:
      List every labeled part, callout, and balloon reference.
      Describe the visible features (e.g., "flange with 8 holes," "reinforced concrete beam," "threaded intake valve").
      Note any technical annotations, material callouts, or treatment requirements (e.g., "Powder Coated," "ISO 2768-m").
    Visual Context & Perspectives:
      Describe the views provided (e.g., isometric, top-down, sectional views A-A, detail zooms).
      Explain the spatial relationship between the main components (e.g., "The motor is mounted via a bracket to the main chassis").
    Key Search Terms for RAG:
      Provide a list of 15-20 technical keywords and synonyms found in or relevant to the drawing. Think like an engineer searching a database (e.g., "Hydraulic assembly," "Steel S355," "Assembly sequence").
    Output Style: Use precise, technical terminology. Prioritize a dense, informative recap of the drawing's content over aesthetic formatting.
"""

ORGANIZE = """
Role: Technical Data Integrator. Task: Consolidate multiple page-by-page descriptions of a technical drawing (File: {filename}) into one single, comprehensive "Master Document."
Core Instruction: Combine all information from the provided page analyses. Do not summarize. Your goal is to ensure that every part number, material, and technical specification mentioned in any page is preserved in this final version.
Structure your response as follows:
- General Document Info: Merge the Title Block data (Title, Drawing Number, Project, and the latest Revision status).
- Unified Master Parts List (BOM): Combine all tables and part lists from all pages into one master list. Include every Part Name, Reference ID, and Material. Ensure no specific part is left out.
- Full Technical Description: Describe the entire assembly by connecting the details from different pages. Explain how the sub-assemblies and components fit together based on the various views provided.
- Annotations & Standards: List all technical notes, manufacturing requirements, and standards (e.g., ISO, DIN, Heat Treatment) found across all pages.
- Search Keywords: Provide a final, exhaustive list of technical terms and keywords extracted from the entire document to optimize search retrieval.
- Constraint: Accuracy and detail are more important than brevity. If a detail was in the page analysis, it must be in this final report.

Input Data: {list_of_page_descriptions}
Output Format: Use Markdown (headings, bullet points, and tables) for a clean, professional structure.
"""


class Draw:
    """
    Convert drawing-style PDFs (technical drawings) into text documents.
    Uses an LLM to produce per-page analyses and can consolidate multiple
    pages into a master document.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
    ):
        """Initialize the Draw converter.

        Args:
            model: LLM model name used for conversion.
        """

        self.llm = ChatOpenAI(model=model)

    @staticmethod
    def page_to_base64(page: fitz.Page) -> str:
        """Render a fitz.Page to a PNG and return its base64 encoding.

        Args:
            page: A single `fitz.Page` object.

        Returns:
            Base64-encoded PNG image of the page.
        """
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_b64

    @staticmethod
    def format_message(query: str, image_str: str) -> list[dict]:
        """Format a message payload containing text and an inline image.

        Args:
            query: Text prompt for the LLM.
            image_str: Base64-encoded image string (may be empty).

        Returns:
            A list representing the message structure expected by the LLM.
        """

        message: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ]

        # Only include the image_url entry when we actually have base64 data.
        if image_str:
            message[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_str}"},
                }
            )

        return message

    def get_document(
        self, filepath: str, page_content: str, doc_type: str, namespace: str
    ) -> list[Document]:
        """Create a `Document` from page content and metadata.

        Args:
            filepath: Original file path used to compute relative media path.
            page_content: Text content for the document.
            doc_type: Document type identifier (e.g., 'draw' or 'text').
            namespace: Namespace to attach to metadata.

        Returns:
            A list with a single `Document` instance.
        """
        return [
            Document(
                page_content=page_content,
                metadata={
                    "path": (
                        Path("media") / Path(filepath).stem / Path(filepath).name
                    ).as_posix(),
                    "page_start": "N/A",
                    "page_end": "N/A",
                    "type": doc_type if doc_type in ["draw", "text"] else "text",
                    "name": Path(filepath).name,
                    "namespace": namespace,
                },
            )
        ]

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a drawing PDF into descriptive documents.

        Each page is analyzed by the LLM. If multiple pages exist, the
        per-page descriptions are consolidated into one master document.

        Args:
            path: Path to the drawing PDF.
            namespace: Namespace to attach to produced documents.

        Returns:
            A list with a single `Document` describing the drawing.
        """
        source = create_source(path)
        copy_source_file(path, source)
        filename = Path(path).stem
        doc = fitz.open(path)
        descriptions = []
        for page in doc:
            image_str = self.page_to_base64(page)
            query = PROMPT.format(filename=filename)
            message = self.format_message(query, image_str)
            response = self.llm.invoke(message)
            descriptions.append(response.content)

        if len(descriptions) == 1:
            page_content = descriptions[0]
            doc_type = "draw"
        else:
            query = ORGANIZE.format(
                filename=filename, list_of_page_descriptions=descriptions
            )
            page_content = self.llm.invoke(self.format_message(query, "")).content
            doc_type = "text"
        return self.get_document(
            filepath=path,
            page_content=str(page_content),
            doc_type=doc_type,
            namespace=namespace,
        )
