import os
import base64
import shutil
import uuid
import traceback
import logging
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from duckling.graph import DucklingGraph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("duckling")

app = FastAPI(title="Duckling Document Extraction Service")
logger.info("Initializing DucklingGraph converter...")
converter = DucklingGraph()
logger.info("DucklingGraph ready.")


@app.post("/convert")
async def convert(
    file: UploadFile = File(...),
):
    request_id = str(uuid.uuid4())
    temp_path = f"temp_{request_id}.pdf"
    archive = None
    folder_path = None
    temp_folder_path = None
    start_time = time.time()

    logger.info(
        "[%s] Received file: %s (content_type=%s)",
        request_id,
        file.filename,
        file.content_type,
    )

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(temp_path)
        logger.info(
            "[%s] Saved temp file: %s (size=%d bytes)", request_id, temp_path, file_size
        )

        namespace = Path(str(file.filename)).stem
        logger.info("[%s] Starting converter for namespace=%s", request_id, namespace)
        state = converter.run(path=temp_path, namespace=namespace)
        logger.info(
            "[%s] Converter finished in %.2fs", request_id, time.time() - start_time
        )

        folder_path = os.path.join("media", namespace)
        os.makedirs(folder_path, exist_ok=True)
        logger.debug("[%s] Output folder: %s", request_id, folder_path)

        temp_folder_path = os.path.join("media", Path(temp_path).stem)
        if os.path.exists(temp_folder_path):
            logger.debug(
                "[%s] Moving artifacts from %s to %s",
                request_id,
                temp_folder_path,
                folder_path,
            )
            for item in os.listdir(temp_folder_path):
                s = os.path.join(temp_folder_path, item)
                d = os.path.join(folder_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            shutil.rmtree(temp_folder_path, ignore_errors=True)

        archive = shutil.make_archive(f"temp_{namespace}", "zip", folder_path)
        archive_size = os.path.getsize(archive)
        logger.info(
            "[%s] Archive created: %s (size=%d bytes)",
            request_id,
            archive,
            archive_size,
        )

        with open(archive, "rb") as f:
            artifacts = base64.b64encode(f.read()).decode()

        elapsed = time.time() - start_time
        logger.info("[%s] Request completed successfully in %.2fs", request_id, elapsed)
        return {"status": "success", "content": state, "artifacts": artifacts}
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(
            "[%s] Unhandled exception after %.2fs: %s\n%s",
            request_id,
            time.time() - start_time,
            e,
            tb,
        )
        raise HTTPException(status_code=500, detail=f"Errore Docling: {str(e)}") from e
    finally:
        for path, label in [
            (temp_path, "temp PDF"),
            (archive, "archive"),
            (temp_folder_path, "temp folder"),
            (folder_path, "output folder"),
        ]:
            if path and os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                    logger.debug("[%s] Cleaned up %s: %s", request_id, label, path)
                except Exception as cleanup_err:
                    logger.warning(
                        "[%s] Failed to clean up %s (%s): %s",
                        request_id,
                        label,
                        path,
                        cleanup_err,
                    )


@app.get("/health")
def health():
    logger.debug("Health check called")
    return {"status": "online"}
