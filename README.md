# 🦆 Duckling

Libreria Python che estrae contenuto testuale e descrizioni di immagini da documenti comuni e restituisce oggetti LangChain `Document` per indicizzazione o RAG.

## Architettura

```
File input → DucklingGraph → docling-serve (parsing/OCR) + OpenAI (immagini/disegni) → Document[]
```

- **Duckling** — routing per formato (LangGraph) e post-processing (chunking, descrizione immagini).
- **docling-serve** — servizio esterno per PDF, tabelle e testo (default: `http://localhost:5001`).
- **OpenAI** — descrizione immagini e pipeline per PDF tecnici/disegno.

Formati supportati: `.pdf`, `.png`/`.jpg`/`.jpeg`, `.csv`/`.xlsx`, `.txt`/`.md`.

## Setup

**1. Dipendenze Python**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
poetry install
```

**2. docling-serve** (Docker)

```powershell
docker run -p 5001:5001 quay.io/docling-project/docling-serve
```

**3. Variabili d'ambiente** (file `.env` nella root del progetto)

```env
OPENAI_API_KEY=sk-...
DOCLING_SERVE_URL=http://localhost:5001   # opzionale
```

## Utilizzo

```python
from duckling.graph import DucklingGraph

graph = DucklingGraph()
state = graph.run(r"C:\path\to\file.pdf", namespace="my-namespace")
documents = state.get("documents", [])
```

I risultati intermedi (JSON, markdown, immagini) vengono salvati in `media/<nome-file>/`.
