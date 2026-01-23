"""Table file converter utilities."""

from duckling.base import BaseConverter
from duckling.utils import copy_source_file, create_source


class Table(BaseConverter):
    """Converter for table files (CSV, XLSX) into text chunks.

    Inherits chunking and loading behavior from `BaseConverter`.
    """

    def __init__(
        self,
        max_tokens: int = 4996,
        tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(max_tokens=max_tokens, tokenizer=tokenizer)

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a table file into a list of document chunks.

        Args:
            path: Path to the table file.
            namespace: Namespace to attach to produced documents.

        Returns:
            A list of chunked `Document` objects.
        """
        source = create_source(path)
        copy_source_file(path, source)
        document = self.load(path)
        chunks = self.chunk(document, namespace=namespace)
        return chunks
