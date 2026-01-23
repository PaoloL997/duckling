"""TypedDict describing runtime state used by the processing graph."""

from typing import TypedDict, List
from pathlib import Path
from langchain_core.documents import Document


class State(TypedDict):
    """Runtime state used by the processing graph.

    Fields:
        input: Path to the input file.
        format: Detected input format (e.g., 'pdf', 'image', 'table').
        namespace: Namespace attached to produced documents.
        documents: List of extracted `Document` objects.
    """

    input: Path
    format: str | None
    namespace: str
    documents: List[Document]
