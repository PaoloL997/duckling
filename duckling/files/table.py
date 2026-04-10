"""Table file converter utilities."""

from duckling.base import BaseConverter
from duckling.utils import copy_source_file, create_source
from duckling.service import LocalService


class Table(BaseConverter):
    """Converter for table files (CSV, XLSX) into text chunks.

    Inherits chunking and loading behavior from `BaseConverter`.
    """

    def __init__(self):
        super().__init__()
        self.service = LocalService()

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a table file into a list of document chunks.

        Args:
            path: Path to the table file.
            namespace: Namespace to attach to produced documents.

        Returns:
            A list of chunked `Document` objects.
        """
        source = create_source(path)
        document = self.service.load_table(path)
        copy_source_file(path, source)
        chunks = self.chunk(document, namespace=namespace)
        return chunks
