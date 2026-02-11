"""Utility helpers for media handling and PDF inspection."""

import shutil
import base64
from pathlib import Path
from langchain_core.documents import Document

import fitz


def copy_source_file(path: str, destination: Path):
    """Copy a source file into a destination directory.

    Args:
        path: Path to the source file.
        destination: Destination directory as a `Path`.
    """
    shutil.copy2(path, destination / Path(path).name)


def file_to_base64(path: str) -> str:
    """Encode a file's contents as a base64 string.

    Args:
        path: Path to the file to encode.

    Returns:
        Base64-encoded string of the file contents.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_source(path: str):
    """Create a media source directory structure for a file.

    The structure created is `media/<stem>/artifacts` and the source
    Path is returned.

    Args:
        path: Original file path used to name the media folder.

    Returns:
        Path to the created source directory.
    """
    root = Path("media")
    root.mkdir(exist_ok=True)
    source = root / Path(path).stem
    source.mkdir(exist_ok=True)
    artifacts = source / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return source


def is_a4(path: str):
    """Check whether a PDF file uses A4 page size within tolerance.

    Args:
        path: Path to the PDF file.

    Returns:
        True if every page matches A4 dimensions (within tolerance), else False.
    """
    a4_width = 595
    a4_height = 842
    tol = 5
    try:
        doc = fitz.open(path)
    except Exception:
        return False
    try:
        if doc.page_count == 0:
            return False
        for p in doc:
            width = p.rect.width
            height = p.rect.height
            portrait_ok = (
                abs(width - a4_width) <= tol and abs(height - a4_height) <= tol
            )
            landscape_ok = (
                abs(width - a4_height) <= tol and abs(height - a4_width) <= tol
            )
            if not (portrait_ok or landscape_ok):
                return False
        return True
    finally:
        doc.close()


def get_types_count(documents: list[Document]) -> dict:
    """Count the number of documents by type.

    Args:
        documents: List of `Document` objects to count.

    Returns:
        Dictionary mapping document types to their counts.
    """
    counts: dict[str, int] = {}
    for doc in documents:
        doc_type = doc.metadata.get("type", "unknown")
        counts[doc_type] = counts.get(doc_type, 0) + 1
    return counts
