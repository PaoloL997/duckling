import os
import subprocess
import sys
import io
import zipfile
import json
from pathlib import Path

import requests

from dotenv import load_dotenv

from docling.datamodel.document import DoclingDocument

load_dotenv()

SERVICE_URL = os.getenv("CLOUD_SERVICE_URL")


class CloudService:
    """Client for Docling-serve cloud service to convert documents via API calls."""

    TO_FORMATS = ["json", "md"]
    IMAGE_EXPORT_MODE = "referenced"
    PIPELINE = "standard"
    OCR_ENGINE = "rapidocr"
    FORCE_OCR = "true"
    TABLE_MODE = "accurate"
    IMAGES_SCALE = "4.0"
    DO_FORMULA_ENRICHMENT = "true"

    def _get_token(self):
        """Obtain an identity token using gcloud CLI for authentication."""
        try:
            gcloud_cmd = "gcloud.cmd" if sys.platform == "win32" else "gcloud"
            token = (
                subprocess.check_output([gcloud_cmd, "auth", "print-identity-token"])
                .decode()
                .strip()
            )
            return token
        except subprocess.CalledProcessError as e:
            print(f"Error obtaining identity token: {e}")
            return None

    def load_pdf(self, path: str):
        """Convert a PDF file to a DoclingDocument using the cloud service."""
        token = self._get_token()
        with open(path, "rb") as f:
            response = requests.post(
                f"{SERVICE_URL}/v1/convert/file",
                headers={"Authorization": f"Bearer {token}"},
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
                timeout=3600,
            )
        response.raise_for_status()
        os.makedirs("media", exist_ok=True)
        filename = Path(path).stem
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(Path("media") / filename)
            json_file = next(f for f in z.namelist() if f.endswith(".json"))
            json_content = json.loads(z.read(json_file))
        return DoclingDocument.model_validate(json_content)

    def load_table(self, path: str):
        """Convert a table file (CSV, XLSX) to a DoclingDocument using the cloud service."""
        token = self._get_token()
        suffix = Path(path).suffix.lower().lstrip(".")
        with open(path, "rb") as f:
            response = requests.post(
                f"{SERVICE_URL}/v1/convert/file",
                headers={"Authorization": f"Bearer {token}"},
                files={"files": (path, f)},
                data={
                    "to_formats": "json",
                    "from_formats": suffix,
                },
                timeout=3600,
            )
        response.raise_for_status()
        json_content = response.json()["document"]["json_content"]
        return DoclingDocument.model_validate(json_content)
