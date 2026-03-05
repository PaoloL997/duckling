import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm
from docling.datamodel.document import DoclingDocument
from dotenv import load_dotenv

load_dotenv()

SERVICE_URL = os.getenv("CLOUD_SERVICE_URL")

# Tuneable constants
LONG_POLL_WAIT_S = 30  # seconds the server holds the connection per poll request
POLL_TIMEOUT_S = 3600  # max total wait time across all polls
WARMUP_TIMEOUT_S = 30  # timeout for the /health warmup ping
MAX_SYNC_RETRIES = 3  # retries for the table (sync) endpoint


class CloudServiceError(RuntimeError):
    """Custom exception for CloudService-related errors."""


class CloudService:
    """Client for Docling-serve (Cloud Run) that uses the *async* conversion
    endpoint to avoid 504 Gateway Timeouts on long-running jobs.

    Flow for PDFs
    ─────────────
    1. POST  /v1/convert/file/async  → task_id
    2. GET   /v1/status/{task_id}    → poll until "success" | "failed"
    3. GET   /v1/result/{task_id}    → download ZIP / JSON

    For short table conversions the sync endpoint is used with retry logic.
    """

    TO_FORMATS = ["json", "md"]
    IMAGE_EXPORT_MODE = "referenced"
    PIPELINE = "standard"
    OCR_ENGINE = "tesseract"
    FORCE_OCR = "true"
    TABLE_MODE = "accurate"
    IMAGES_SCALE = "4.0"
    DO_FORMULA_ENRICHMENT = "true"

    def _get_token(self) -> str:
        """Return a fresh GCP identity token via gcloud CLI."""
        try:
            gcloud_cmd = "gcloud.cmd" if sys.platform == "win32" else "gcloud"
            return (
                subprocess.check_output([gcloud_cmd, "auth", "print-identity-token"])
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as e:
            raise CloudServiceError(f"Cannot obtain identity token: {e}") from e

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def warmup(self) -> bool:
        """Ping /health to wake the container before a heavy job.

        Returns True if the service is ready, False on timeout/error.
        """
        url = f"{SERVICE_URL}/health"
        print("Warming up Cloud Run container …", end=" ", flush=True)
        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                timeout=WARMUP_TIMEOUT_S,
            )
            resp.raise_for_status()
            print("ready ✓")
            return True
        except requests.RequestException as exc:
            print(f"warmup failed ({exc})")
            return False

    def _submit_async(self, path: str) -> str:
        """POST to the async endpoint and return the task_id."""
        with open(path, "rb") as f:
            response = requests.post(
                f"{SERVICE_URL}/v1/convert/file/async",
                headers=self._headers(),
                files={"files": (path, f, "application/pdf")},
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
                },
                timeout=60,  # only the *submission* needs a short timeout
            )
        response.raise_for_status()
        task_id = response.json().get("task_id")
        if not task_id:
            raise CloudServiceError(f"No task_id in response: {response.text}")
        print(f"Task submitted → id={task_id}")
        return task_id

    def _poll_until_done(self, task_id: str) -> None:
        """Block until the task reaches 'success' using server-side long-polling.

        The `wait` parameter tells the server to hold the connection open for up to
        LONG_POLL_WAIT_S seconds and return immediately when the status changes.
        This avoids a busy loop and reduces the total number of HTTP calls to ~1
        per LONG_POLL_WAIT_S seconds.
        """
        deadline = time.monotonic() + POLL_TIMEOUT_S
        progress: tqdm | None = None

        try:
            while time.monotonic() < deadline:
                resp = requests.get(
                    f"{SERVICE_URL}/v1/status/poll/{task_id}",
                    headers=self._headers(),
                    params={"wait": LONG_POLL_WAIT_S},
                    timeout=LONG_POLL_WAIT_S + 10,
                )
                resp.raise_for_status()
                body = resp.json()
                task_status = body.get("task_status", "unknown")
                meta = body.get("task_meta") or {}

                num_docs = meta.get("num_docs")
                num_processed = meta.get("num_processed", 0)

                # Initialise the bar as soon as we know the total
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
                    raise CloudServiceError(
                        f"Task {task_id} failed: {body.get('error_message')}"
                    )
        finally:
            if progress is not None:
                progress.close()

        raise CloudServiceError(
            f"Task {task_id} did not complete within {POLL_TIMEOUT_S}s"
        )

    def _download_result(self, task_id: str) -> bytes:
        """Fetch the ZIP result for a completed task."""
        resp = requests.get(
            f"{SERVICE_URL}/v1/result/{task_id}",
            headers=self._headers(),
            timeout=120,
        )
        resp.raise_for_status()
        return resp.content

    def load_pdf(self, path: str) -> DoclingDocument:
        """Convert a PDF → DoclingDocument using the async endpoint."""
        self.warmup()

        task_id = self._submit_async(path)
        self._poll_until_done(task_id)
        zip_bytes = self._download_result(task_id)

        os.makedirs("media", exist_ok=True)
        filename = Path(path).stem
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(Path("media") / filename)
            json_file = next(f for f in z.namelist() if f.endswith(".json"))
            json_content = json.loads(z.read(json_file))

        return DoclingDocument.model_validate(json_content)

    def load_table(self, path: str) -> DoclingDocument:
        """Convert a CSV/XLSX → DoclingDocument (sync with retry)."""
        suffix = Path(path).suffix.lower().lstrip(".")
        last_exc: Exception | None = None

        for attempt in range(1, MAX_SYNC_RETRIES + 1):
            try:
                with open(path, "rb") as f:
                    response = requests.post(
                        f"{SERVICE_URL}/v1/convert/file",
                        headers=self._headers(),
                        files={"files": (path, f)},
                        data={"to_formats": "json", "from_formats": suffix},
                        timeout=300,
                    )
                response.raise_for_status()
                json_content = response.json()["document"]["json_content"]
                return DoclingDocument.model_validate(json_content)

            except requests.RequestException as exc:
                last_exc = exc
                wait = 2**attempt  # 2 s, 4 s, 8 s
                print(
                    f"load_table attempt {attempt}/{MAX_SYNC_RETRIES} failed "
                    f"({exc}). Retrying in {wait}s …"
                )
                time.sleep(wait)

        raise CloudServiceError(
            f"load_table failed after {MAX_SYNC_RETRIES} attempts"
        ) from last_exc
