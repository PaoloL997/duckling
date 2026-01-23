# 🦆 Duckling

What it does
------------
Duckling extracts textual content and image descriptions from common
document formats (PDF, images, CSV/XLSX) and returns them as
standardized LangChain `Document` objects for downstream indexing or
retrieval-augmented generation.

How it works
-------------------------
- Detects the input file format and dispatches to a dedicated converter.
- PDF converter uses Docling for parsing and OCR, extracting text and
- Drawing PDFs are handled by a drawing-focused pipeline that prompts
- Images are encoded and described via an LLM prompt; tables are
 
Technologies
---------------------------------
- Docling / docling_core: PDF parsing, OCR and image artifact extraction.
- LangChain / langchain_core: Standard `Document` model used as output.
- LangGraph: Small state graph to route files by detected format.
- OpenAI-compatible LLM (via `langchain_openai.ChatOpenAI`): image and
	drawing description prompts and refinement.
- PyMuPDF (`fitz`) and OpenCV (`cv2`): page rendering and image handling.
- Transformers (HuggingFace tokenizer): token-aware chunking for text.

Minimal example
---------------
1. Create and activate a virtual environment and install dependencies:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
poetry install
```

2. Ensure LLM credentials and any environment settings are available
	 (for example, place keys in a `.env` file read by the app).

3. Example usage (Python):

```python
from duckling.graph import DucklingGraph

graph = DucklingGraph()
state = graph.run(r"C:\path\to\file.pdf", namespace="my-namespace")
documents = state.get("documents", [])
```
