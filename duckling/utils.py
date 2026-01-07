import statistics
import fitz


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
