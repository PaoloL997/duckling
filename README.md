# Process Docs

PDF document processing using Docling and OpenAI for structured content extraction.

## Installation

```bash
# Install dependencies
poetry install

# Set up environment
echo "OPENAI_API_KEY=your_api_key" > .env
```

## Usage

```python
from process_docs.convert import DocumentProcessor

# Initialize processor
processor = DocumentProcessor()

# Process document
docs = processor.process("document.pdf")
```

## Input

- **PDF file path** (string): Path to the PDF document to process
- **Output path** (optional): Path for markdown output (default: "output.md")

## Output

Returns a list of `langchain.schema.Document` objects containing both text chunks and image descriptions.

Each Document contains:
- `page_content`: The actual content (text or image description)
- `metadata`: Dictionary with document information

### Text Documents
```python
Document(
    page_content="Extracted text content from PDF...",
    metadata={
        "path": "document.pdf",           # Source PDF file path
        "pages": [1, 3],                  # Page range [start, end]
        "type": "text",                   # Content type
        "name": None                      # No specific name for text chunks
    }
)
```

### Image Documents  
```python
Document(
    page_content="AI-generated detailed description of the image...",
    metadata={
        "path": "output_artifacts/figure_1.png",  # Generated image file path
        "pages": None,                            # No specific page range
        "type": "image",                          # Content type
        "name": "Figure 1"                       # Image reference name from document
    }
)
```

### Metadata Fields
- **path**: File path (source PDF for text, generated image path for images)
- **pages**: Page range as `[start_page, end_page]` for text chunks, `None` for images
- **type**: Content type (`"text"` or `"image"`)
- **name**: Reference name from document (e.g., "Figure 1", "Table 2") or `None` for text

