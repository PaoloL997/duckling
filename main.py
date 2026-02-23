import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse

from duckling.graph import DucklingGraph


app = FastAPI(title="Duckling Document Extraction Service")


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
    max_tokens: int = Query(4096),
    tokenizer: str = Query("sentence-transformers/all-MiniLM-L6-v2"),
    llm: str = Query("gpt-4.1-nano"),
):
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        converter = DucklingGraph(max_tokens=max_tokens, tokenizer=tokenizer, llm=llm)
        state = converter.run(path=temp_path, namespace=str(file.filename))
        return {"status": "success", "content": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore Docling: {str(e)}") from e

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/artifacts/{namespace}")
def download(namespace: str, background_tasks: BackgroundTasks):
    folder_path = os.path.join("media", namespace)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Namespace non trovato")

    zip_path = f"temp_{namespace}"
    try:
        archive = shutil.make_archive(zip_path, "zip", folder_path)
        background_tasks.add_task(os.remove, archive)
        return FileResponse(
            path=archive, filename=f"{namespace}.zip", media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore creazione archivio: {str(e)}"
        ) from e


@app.get("/health")
def health():
    return {"status": "online"}
