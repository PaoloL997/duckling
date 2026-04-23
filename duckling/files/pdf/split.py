"""PDF splitting utilities used to avoid OOM on very large documents."""

from pathlib import Path
from typing import List, Tuple

import fitz


DEFAULT_PAGES_PER_CHUNK = 50


def _top_level_toc_ranges(
    doc: "fitz.Document",
) -> List[Tuple[int, int, str]]:
    """Return top-level TOC ranges as ``(start, end, title)`` tuples.

    Page indices are zero-based and inclusive. If the first chapter does
    not start at page 0, a leading "front_matter" range is prepended to
    preserve every page of the original document.
    """
    toc = doc.get_toc() or []
    entries = [
        (title, max(0, page_no - 1))
        for level, title, page_no in toc
        if level == 1 and page_no and page_no > 0
    ]
    if not entries:
        return []

    total = doc.page_count
    ranges: List[Tuple[int, int, str]] = []

    if entries[0][1] > 0:
        ranges.append((0, entries[0][1] - 1, "front_matter"))

    for i, (title, start) in enumerate(entries):
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else total - 1
        if end < start:
            end = start
        ranges.append((start, end, title))

    return ranges


def _fixed_size_ranges(
    total_pages: int, pages_per_chunk: int
) -> List[Tuple[int, int, str]]:
    """Return fixed-size ranges covering ``total_pages``."""
    ranges: List[Tuple[int, int, str]] = []
    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk - 1, total_pages - 1)
        ranges.append((start, end, f"pages_{start + 1}_{end + 1}"))
    return ranges


def split_pdf(
    path: str,
    output_dir: str,
    pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK,
) -> List[Tuple[str, int]]:
    """Split a PDF into smaller files by top-level TOC or fixed-size chunks.

    Args:
        path: Input PDF file.
        output_dir: Directory where chunk PDFs will be written.
        pages_per_chunk: Fallback chunk size used when no TOC is available.

    Returns:
        A list of ``(chunk_path, page_offset)`` tuples, where
        ``page_offset`` is the zero-based index of the first page of the
        chunk within the original document.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem

    doc = fitz.open(path)
    try:
        total = doc.page_count
        if total == 0:
            return []

        ranges = _top_level_toc_ranges(doc)
        if not ranges:
            ranges = _fixed_size_ranges(total, pages_per_chunk)

        chunk_files: List[Tuple[str, int]] = []
        for idx, (start, end, _title) in enumerate(ranges, start=1):
            chunk_path = out / f"{stem}__chunk_{idx:04d}.pdf"
            chunk_doc = fitz.open()
            try:
                chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
                chunk_doc.save(str(chunk_path))
            finally:
                chunk_doc.close()
            chunk_files.append((str(chunk_path), start))

        return chunk_files
    finally:
        doc.close()
