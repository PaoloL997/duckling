import os
import base64
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from duckling.graph import DucklingGraph

app = FastAPI(title="Duckling Document Extraction Service")
converter = DucklingGraph()  # TODO: converter eseguito solo all'inizio


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
):
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        namespace = Path(str(file.filename)).stem
        state = converter.run(path=temp_path, namespace=namespace)

        folder_path = os.path.join("media", namespace)
        zip_path = f"temp_{namespace}"
        archive = shutil.make_archive(zip_path, "zip", folder_path)

        with open(archive, "rb") as f:
            zip_data = f.read()

        artifacts = base64.b64encode(zip_data).decode()
        os.remove(archive)
        shutil.rmtree(folder_path, ignore_errors=True)

        return {"status": "success", "content": state, "artifacts": artifacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore Docling: {str(e)}") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/health")
def health():
    return {"status": "online"}
