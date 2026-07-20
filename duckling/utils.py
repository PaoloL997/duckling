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


def _matches_page_size(
    width: float,
    height: float,
    target_width: float,
    target_height: float,
    tol: float,
) -> bool:
    """Return True when a page matches target size in portrait or landscape."""
    portrait_ok = (
        abs(width - target_width) <= tol and abs(height - target_height) <= tol
    )
    landscape_ok = (
        abs(width - target_height) <= tol and abs(height - target_width) <= tol
    )
    return portrait_ok or landscape_ok


def is_standard_pdf_layout(
    path: str,
    tol: float = 8.0,
    min_standard_page_ratio: float = 0.7,
) -> bool:
    """Heuristic to decide if a PDF should go through standard text extraction.

    A page is considered standard if it is close to A4 or US Letter
    (portrait or landscape). The file is considered standard when at least
    ``min_standard_page_ratio`` of its pages match one of these sizes.

    Args:
        path: Path to the PDF file.
        tol: Allowed absolute tolerance on width/height in points.
        min_standard_page_ratio: Minimum ratio of standard-size pages.

    Returns:
        True if the PDF is mostly standard-size pages, else False.
    """
    a4_width = 595.0
    a4_height = 842.0
    letter_width = 612.0
    letter_height = 792.0
    try:
        doc = fitz.open(path)
    except Exception:
        return False
    try:
        if doc.page_count == 0:
            return False

        standard_pages = 0
        for page in doc:
            width = page.rect.width
            height = page.rect.height
            matches_a4 = _matches_page_size(width, height, a4_width, a4_height, tol)
            matches_letter = _matches_page_size(
                width, height, letter_width, letter_height, tol
            )
            if matches_a4 or matches_letter:
                standard_pages += 1

        ratio = standard_pages / doc.page_count
        return ratio >= min_standard_page_ratio
    finally:
        doc.close()


def is_a4(path: str):
    """Backward compatible wrapper for older callers.

    Historically this function required every page to be A4. It now reuses
    the standard-layout heuristic to avoid misrouting normal US Letter PDFs
    to drawing conversion.
    """
    return is_standard_pdf_layout(path=path)


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
