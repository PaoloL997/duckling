"""Routing graph that dispatches files to appropriate converters."""

from pathlib import Path

from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from langgraph.graph import StateGraph, END

from duckling.files.table import Table
from duckling.files.image import Image
from duckling.files.pdf.convert import PDF
from duckling.files.pdf.draw import Draw
from duckling.utils import is_a4, get_types_count
from duckling.state import State


ACCEPTED_FORMATS = {
    "image": [".png", ".jpg", ".jpeg"],
    "table": [".csv", ".xlsx"],
    "pdf": [".pdf"],
}


class DucklingGraph:
    """Routing graph that converts different file types into documents.

    The graph inspects an input file, determines its format (pdf, image, table),
    and dispatches to the appropriate converter pipeline.
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm: str = "gpt-4.1-nano",
    ):
        """Initialize the DucklingGraph.

        Args:
            max_tokens: Token limit used by downstream converters.
            tokenizer: Tokenizer identifier.
            llm: LLM model name used by converters.
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.llm = llm
        self._warmup()
        self.graph = self._compile()

    def _warmup(self) -> None:
        """Pre-load and cache all models required at runtime.

        This method is intentionally called during __init__ so that the Docker
        build step ``RUN python -c "from duckling.graph import DucklingGraph;
        DucklingGraph()"`` triggers all HuggingFace / Docling downloads and
        saves them to the local cache.  The container can then run fully
        offline (``HF_HUB_OFFLINE=1``) without any network access.
        """
        print("[DucklingGraph] Warming up models...")

        # 1. HuggingFace tokenizer (used by BaseConverter / HybridChunker)
        print("[DucklingGraph]   - HuggingFace tokenizer")
        AutoTokenizer.from_pretrained(self.tokenizer)

        # 2. Docling DocumentConverter with full PDF pipeline
        #    (downloads layout, OCR and formula-enrichment models)
        print("[DucklingGraph]   - Docling PDF pipeline")
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=True,
            do_formula_enrichment=True,
            images_scale=4,
        )
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA, num_threads=8
        )
        pipeline_options.do_ocr = True
        DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        print("[DucklingGraph] Warmup complete.")

    def _format_node(self, state: State) -> dict:
        """Determine the format of the input file.

        Args:
            state: Current state containing an `input` Path.

        Returns:
            A dict with the detected `format` key.
        """
        path = state["input"]
        extension = path.suffix.lower()

        for fmt, extensions in ACCEPTED_FORMATS.items():
            if extension in extensions:
                return {"format": fmt}

        raise ValueError(f"File format not supported: {extension}")

    def _pdf(self, _state: State) -> dict:
        """PDF routing node (does not modify state).

        Returns an empty dict to continue graph execution.
        """
        return {}

    def _standard_pdf(self, state: State) -> dict:
        """Convert a standard A4 PDF into documents.

        Args:
            state: Current processing state.

        Returns:
            Dict containing `documents` extracted from the PDF.
        """
        converter = PDF(
            max_tokens=self.max_tokens, tokenizer=self.tokenizer, model=self.llm
        )
        documents = converter.convert(
            path=str(state["input"]), namespace=state["namespace"]
        )
        return {"documents": documents}

    def _drawing_pdf(self, state: State) -> dict:
        """Convert a drawing-style PDF using drawing-based converter.

        Args:
            state: Current processing state.

        Returns:
            Dict containing `documents` extracted from the drawing-based conversion.
        """
        draw = Draw(model=self.llm)
        documents = draw.convert(path=str(state["input"]), namespace=state["namespace"])
        return {"documents": documents}

    def _check_documents(self, state: State) -> dict:
        """Check whether documents were extracted and optionally log fallback.

        Args:
            state: Current processing state.

        Returns:
            Empty dict; the graph uses conditional edges to decide next step.
        """
        types_count = get_types_count(state.get("documents", []))
        print(f"Document types count: {types_count}")

        # Check if we have text documents - if not, we'll fallback to drawing PDF
        if len(state.get("documents", [])) == 0 or types_count.get("text", 0) == 0:
            print(
                "No text documents extracted from standard PDF conversion. Falling back to drawing-based conversion."
            )
        return {}

    def _image(self, state: State) -> dict:
        """Convert an image file into a descriptive document.

        Args:
            state: Current processing state.

        Returns:
            Dict containing `documents` produced from the image.
        """
        converter = Image(model=self.llm)
        documents = converter.convert(
            path=str(state["input"]), namespace=state["namespace"]
        )
        return {"documents": documents}

    def _table(self, state: State) -> dict:
        """Convert a table file (CSV/XLSX) into text documents.

        Args:
            state: Current processing state.

        Returns:
            Dict containing `documents` extracted from the table.
        """
        converter = Table(max_tokens=self.max_tokens, tokenizer=self.tokenizer)
        documents = converter.convert(
            path=str(state["input"]), namespace=state["namespace"]
        )
        return {"documents": documents}

    def _compile(self):
        """Build and compile the internal StateGraph.

        Returns:
            A compiled graph ready for invocation.
        """
        graph = StateGraph(State)

        graph.add_node("format", self._format_node)
        graph.add_node("pdf", self._pdf)
        graph.add_node("standard_pdf", self._standard_pdf)
        graph.add_node("check_documents", self._check_documents)
        graph.add_node("drawing_pdf", self._drawing_pdf)
        graph.add_node("image", self._image)
        graph.add_node("table", self._table)

        graph.set_entry_point("format")

        graph.add_conditional_edges(
            "format",
            lambda state: state["format"],
            {"pdf": "pdf", "image": "image", "table": "table"},
        )

        graph.add_conditional_edges(
            "pdf",
            lambda state: "standard" if is_a4(state["input"]) else "other",
            {"standard": "standard_pdf", "other": "drawing_pdf"},
        )

        graph.add_edge("standard_pdf", "check_documents")
        graph.add_edge("drawing_pdf", END)

        # Nodo di controllo: se non ci sono documenti di tipo text -> fallback su drawing_pdf
        def should_fallback(state):
            """Check if we should fallback to drawing PDF conversion."""
            types_count = get_types_count(state.get("documents", []))
            has_text_documents = types_count.get("text", 0) > 0
            has_any_documents = len(state.get("documents", [])) > 0

            if has_any_documents and has_text_documents:
                return "has_documents"
            return "no_documents"

        graph.add_conditional_edges(
            "check_documents",
            should_fallback,
            {"has_documents": END, "no_documents": "drawing_pdf"},
        )
        graph.add_edge("image", END)
        graph.add_edge("table", END)

        return graph.compile()

    def run(self, path: str, namespace: str = "namespace") -> State:
        """Run the graph starting from an input path.

        Args:
            path: Input file path to process.
            namespace: Namespace to attach to produced documents.

        Returns:
            Final `State` after graph execution.
        """
        initial_state: State = {
            "input": Path(path),
            "format": None,
            "namespace": namespace,
            "documents": [],
        }

        return self.graph.invoke(initial_state)
