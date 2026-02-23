import os
import shutil
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from duckling.graph import DucklingGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"Ricevuto file: {file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File salvato in: {temp_path}, dimensione: {os.path.getsize(temp_path)} bytes")
        namespace = Path(file.filename).stem
        logger.info(f"Namespace: {namespace}")
        converter = DucklingGraph(max_tokens=max_tokens, tokenizer=tokenizer, llm=llm)
        logger.info("DucklingGraph inizializzato, avvio conversione...")
        state = converter.run(path=temp_path, namespace=namespace)
        logger.info("Conversione completata!")
        return {"status": "success", "content": state}
    except Exception as e:
        logger.error(f"Errore: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore Docling: {str(e)}") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"File temporaneo rimosso: {temp_path}")

@app.get("/artifacts/{dir}")
def download(dir: str, background_tasks: BackgroundTasks):
    folder_path = os.path.join("media", dir)
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=404, detail="Directory non trovata")
    zip_path = f"temp_{dir}"
    try:
        archive = shutil.make_archive(zip_path, "zip", folder_path)
        background_tasks.add_task(os.remove, archive)
        background_tasks.add_task(shutil.rmtree, folder_path)
        return FileResponse(
            path=archive, filename=f"{dir}.zip", media_type="application/zip"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore creazione archivio: {str(e)}"
        ) from e

@app.get("/health")
def health():
    return {"status": "online"}
