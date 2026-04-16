from duckling.base import BaseConverter
from duckling.utils import copy_source_file, create_source
from duckling.service import LocalService


class Text(BaseConverter):
    """Converter for text (TXT/MD) files."""

    def __init__(self):
        super().__init__()
        self.service = LocalService()

    def convert(self, path: str, namespace: str = "namespace"):
        """Convert a text file into a list of document chunks.

        Args:
            path: Path to the text file.
            namespace: Namespace to attach to produced documents.
        """
        source = create_source(path)
        document = self.service.load_textual(path)
        copy_source_file(path, source)
        return self.chunk(document, namespace=namespace)
