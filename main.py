import os
import base64
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from duckling.graph import DucklingGraph

app = FastAPI(title="Duckling Document Extraction Service")
converter = DucklingGraph()


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
):
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    folder_path = None
    archive = None
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        namespace = Path(str(file.filename)).stem
        state = converter.run(path=temp_path, namespace=namespace)

        folder_path = os.path.join("media", namespace)
        os.makedirs(folder_path, exist_ok=True)

        temp_folder_path = os.path.join("media", Path(temp_path).stem)
        if os.path.exists(temp_folder_path):
            for item in os.listdir(temp_folder_path):
                s = os.path.join(temp_folder_path, item)
                d = os.path.join(folder_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(temp_folder_path, ignore_errors=True)

        archive = shutil.make_archive(f"temp_{namespace}", "zip", folder_path)

        with open(archive, "rb") as f:
            artifacts = base64.b64encode(f.read()).decode()

        return {"status": "success", "content": state, "artifacts": artifacts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore Docling: {str(e)}") from e
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if archive and os.path.exists(archive):
            os.remove(archive)
        if folder_path and os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "online"}
