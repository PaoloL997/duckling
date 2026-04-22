"""Client for a local docling-serve container."""

import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
from docling.datamodel.document import DoclingDocument
from dotenv import load_dotenv

load_dotenv()

SERVICE_URL = os.getenv("DOCLING_SERVE_URL", "http://localhost:5001")

# Tuneable constants
LONG_POLL_WAIT_S = 30  # seconds the server holds the connection per poll request
POLL_TIMEOUT_S = 3600  # max total wait time across all polls
HEALTHCHECK_TIMEOUT_S = 10  # timeout for the /health ping
MAX_SYNC_RETRIES = 3  # retries for the table (sync) endpoint


class LocalServiceError(RuntimeError):
    """Custom exception for LocalService-related errors."""


class LocalService:
    """Client for a local docling-serve container.

    The container is expected to be running at *SERVICE_URL*
    (default ``http://localhost:5001``).

    Flow for PDFs
    ─────────────
    1. POST  /v1/convert/file/async  → task_id
    2. GET   /v1/status/poll/{task_id}  → poll until "success" | "failure"
    3. GET   /v1/result/{task_id}    → download ZIP / JSON

    For short table conversions the sync endpoint is used with retry logic.
    """

    TO_FORMATS = ["json", "md"]
    IMAGE_EXPORT_MODE = "referenced"
    PIPELINE = "standard"
    OCR_ENGINE = "easyocr"
    FORCE_OCR = "true"
    TABLE_MODE = "accurate"
    IMAGES_SCALE = "4.0"
    DO_FORMULA_ENRICHMENT = "true"

    @staticmethod
    def _validate_document(json_content: dict) -> DoclingDocument:
        """Validate a Docling JSON payload against the local SDK schema.

        The local environment currently uses schema 1.9.0, while the service
        can return 1.10.0 for markdown conversions. Normalizing the top-level
        version keeps the payload compatible with the installed SDK.
        """
        normalized = dict(json_content)
        if normalized.get("version") == "1.10.0":
            normalized["version"] = "1.9.0"
        return DoclingDocument.model_validate(normalized)

    def healthcheck(self) -> bool:
        """Ping /health to verify the local container is reachable.

        Returns:
            True if the service is healthy, False otherwise.
        """
        url = f"{SERVICE_URL}/health"
        print("Checking local docling-serve...", end=" ", flush=True)
        try:
            resp = requests.get(url, timeout=HEALTHCHECK_TIMEOUT_S)
            resp.raise_for_status()
            print("ready")
            return True
        except requests.RequestException as exc:
            print(f"healthcheck failed ({exc})")
            return False

    def _submit_async(self, path: str) -> str:
        """POST to the async endpoint and return the task_id."""
        with open(path, "rb") as f:
            response = requests.post(
                f"{SERVICE_URL}/v1/convert/file/async",
                files={"files": (Path(path).name, f, "application/pdf")},
                data={
                    "target_type": "zip",
                    "to_formats": self.TO_FORMATS,
                    "image_export_mode": self.IMAGE_EXPORT_MODE,
                    "pipeline": self.PIPELINE,
                    "ocr_engine": self.OCR_ENGINE,
                    "force_ocr": self.FORCE_OCR,
                    "table_mode": self.TABLE_MODE,
                    "images_scale": self.IMAGES_SCALE,
                    "do_formula_enrichment": self.DO_FORMULA_ENRICHMENT,
                    "num_threads": 8,
                },
                timeout=60,
            )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise LocalServiceError(f"No task_id in response: {response.text}")
        print(f"Task submitted → id={task_id}")
        return task_id

    def _poll_until_done(self, task_id: str) -> None:
        """Block until the task reaches 'success' using server-side long-polling."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        progress: tqdm | None = None

        try:
            while time.monotonic() < deadline:
                resp = requests.get(
                    f"{SERVICE_URL}/v1/status/poll/{task_id}",
                    params={"wait": LONG_POLL_WAIT_S},
                    timeout=LONG_POLL_WAIT_S + 10,
                )
                resp.raise_for_status()
                body = resp.json()
                task_status = body.get("task_status", "unknown")
                meta = body.get("task_meta") or {}

                num_docs = meta.get("num_docs")
                num_processed = meta.get("num_processed", 0)

                if progress is None and num_docs:
                    progress = tqdm(
                        total=num_docs,
                        desc="Converting",
                        unit="page",
                        dynamic_ncols=True,
                    )

                if progress is not None:
                    progress.n = num_processed
                    progress.set_postfix(status=task_status)
                    progress.refresh()

                if task_status == "success":
                    if progress is not None:
                        progress.n = progress.total
                        progress.refresh()
                    return
                if task_status == "failure":
                    raise LocalServiceError(
                        f"Task {task_id} failed: {body.get('error_message')}"
                    )
        finally:
            if progress is not None:
                progress.close()

        raise LocalServiceError(
            f"Task {task_id} did not complete within {POLL_TIMEOUT_S}s"
        )

    def _download_result(self, task_id: str) -> bytes:
        """Fetch the ZIP result for a completed task."""
        resp = requests.get(
            f"{SERVICE_URL}/v1/result/{task_id}",
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content

    def load_pdf(self, path: str) -> DoclingDocument:
        """Convert a PDF → DoclingDocument using the async endpoint."""
        self.healthcheck()

        task_id = self._submit_async(path)
        self._poll_until_done(task_id)
        zip_bytes = self._download_result(task_id)

        os.makedirs("media", exist_ok=True)
        filename = Path(path).stem.strip()
        target_dir = Path("media") / filename
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(target_dir)
            json_file = next(f for f in z.namelist() if f.endswith(".json"))
            json_content = json.loads(z.read(json_file))

        return self._validate_document(json_content)

    def load_table(self, path: str) -> DoclingDocument:
        """Convert a CSV/XLSX → DoclingDocument (sync with retry)."""
        suffix = Path(path).suffix.lower().lstrip(".")
        last_exc: Exception | None = None

        for attempt in range(1, MAX_SYNC_RETRIES + 1):
            try:
                with open(path, "rb") as f:
                    response = requests.post(
                        f"{SERVICE_URL}/v1/convert/file",
                        files={"files": (Path(path).name, f)},
                        data={"to_formats": "json", "from_formats": suffix},
                        timeout=300,
                    )
                response.raise_for_status()
                json_content = response.json()["document"]["json_content"]
                return self._validate_document(json_content)

            except requests.RequestException as exc:
                last_exc = exc
                wait = 2**attempt  # 2 s, 4 s, 8 s
                print(
                    f"load_table attempt {attempt}/{MAX_SYNC_RETRIES} failed "
                    f"({exc}). Retrying in {wait}s …"
                )
                time.sleep(wait)

        raise LocalServiceError(
            f"load_table failed after {MAX_SYNC_RETRIES} attempts"
        ) from last_exc

    def load_textual(self, path: str) -> DoclingDocument:
        """Convert a TXT/MD file via the local service sync endpoint.

        The backend supports markdown but not plain txt, so txt files are sent
        as markdown-compatible plain text content.
        """
        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix not in ["txt", "md"]:
            raise LocalServiceError(f"Unsupported text file format: {suffix}")

        self.healthcheck()

        from_format = "md"
        mime_type = "text/markdown"
        filename = Path(path).name if suffix == "md" else f"{Path(path).stem}.md"

        with open(path, "rb") as f:
            payload = f.read()

        with io.BytesIO(payload) as file_obj:
            response = requests.post(
                f"{SERVICE_URL}/v1/convert/file",
                files={"files": (filename, file_obj, mime_type)},
                data={"to_formats": "json", "from_formats": from_format},
                timeout=120,
            )
        response.raise_for_status()
        json_content = response.json()["document"]["json_content"]
        return self._validate_document(json_content)
