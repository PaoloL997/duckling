import os
import re
from pathlib import Path
import cv2


class ImageOptions:
    """Filter images in markdown based on their dimensions."""

    def __init__(
        self,
        page_width: int,
        page_height: int,
        min_size_ratio: float = 0.10,
    ):
        """Initialize ImageOptions with page dimensions and minimum size ratio.

        Args:
            page_width: Width of the page in pixels
            page_height: Height of the page in pixels
            min_size_ratio: Minimum size ratio to determine if an image should be kept
        """
        self.page_width = page_width
        self.page_height = page_height
        self.min_size_ratio = min_size_ratio

    def find_images(self, content: str):
        """Find all image paths in the markdown content.

        Args:
            content: Markdown content as a string
        """
        pattern = r"!\[.*?\]\((.*?)\)"
        return re.findall(pattern, content)

    def get_image_dimensions(self, path: str):
        """Get dimensions of an image given its file path.
        Args:
            path: File path to the image
        """
        img = cv2.imread(path)
        if img is not None:
            height, width = img.shape[:2]
            return width, height
        return None, None

    def _find_image_references(self, content: str):
        """Find all image references in the markdown content."""
        image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
        return re.findall(image_pattern, content)

    def _should_remove_image(self, img_path: str, source_path: Path):
        """Check if an image should be removed based on its dimensions."""
        full_img_path = source_path / img_path

        if not full_img_path.exists():
            return True

        width, height = self.get_image_dimensions(str(full_img_path))

        if width is None or height is None:
            return True

        min_width = self.page_width * self.min_size_ratio
        min_height = self.page_height * self.min_size_ratio

        return width < min_width or height < min_height

    def _remove_image_file(self, img_path: str, source_path: Path):
        """Remove image file from disk."""

        try:
            full_img_path = source_path / img_path
            os.remove(str(full_img_path))
            return True
        except Exception:
            return False

    def _remove_markdown_reference(
        self, markdown_content: str, alt_text: str, img_path: str
    ):
        """Remove the image reference from the markdown content."""
        img_reference = f"![{alt_text}]({img_path})"
        return markdown_content.replace(img_reference, "")

    def _clean_markdown_formatting(self, markdown_content: str):
        """Clean the markdown formatting by removing multiple empty lines"""
        return re.sub(r"\n\s*\n\s*\n", "\n\n", markdown_content)

    def filter_images(self, markdown_content: str, source_path: Path):
        """
        Filter images based on their dimensions, physically remove those that are too small,
        and remove their references from the markdown content.

        Args:
            markdown_content: Markdown content with image references
            source_path: Base path where images are located
        """
        matches = self._find_image_references(markdown_content)
        cleaned_markdown = markdown_content

        for alt_text, img_path in matches:
            if self._should_remove_image(img_path, source_path):
                self._remove_image_file(img_path, source_path)
                cleaned_markdown = self._remove_markdown_reference(
                    cleaned_markdown, alt_text, img_path
                )

        return self._clean_markdown_formatting(cleaned_markdown)
