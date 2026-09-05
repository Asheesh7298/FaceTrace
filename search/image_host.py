"""
search/image_host.py
====================
Uploads a local image to a public URL so URL-only reverse-image engines
(Google Lens via SerpApi) can read it.

Primary host : catbox.moe  (free, no API key, permanent unlisted URL)
Fallback host: 0x0.st      (free, no API key, temporary)

The uploaded crop is the *face crop*, not the original photo. For a face
identification pipeline this is an outward-facing action — see README
"Privacy & limitations".
"""

import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CATBOX_API = "https://catbox.moe/user/api.php"
NULL_POINTER = "https://0x0.st"

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceTrace/1.0"


def _upload_catbox(path: str, timeout: int = 30) -> Optional[str]:
    with open(path, "rb") as f:
        resp = requests.post(
            CATBOX_API,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (Path(path).name, f)},
            headers={"User-Agent": _UA},
            timeout=timeout,
        )
    if resp.status_code == 200 and resp.text.strip().startswith("http"):
        return resp.text.strip()
    logger.warning(f"catbox upload failed: {resp.status_code} {resp.text[:80]}")
    return None


def _upload_0x0(path: str, timeout: int = 30) -> Optional[str]:
    with open(path, "rb") as f:
        resp = requests.post(
            NULL_POINTER,
            files={"file": (Path(path).name, f)},
            headers={"User-Agent": _UA},
            timeout=timeout,
        )
    if resp.status_code == 200 and resp.text.strip().startswith("http"):
        return resp.text.strip()
    logger.warning(f"0x0.st upload failed: {resp.status_code} {resp.text[:80]}")
    return None


def host_image(path: str) -> Optional[str]:
    """
    Upload `path` to a public host and return its URL, or None if all hosts fail.
    Tries catbox first, then 0x0.st.
    """
    path = str(Path(path).resolve())
    for name, fn in (("catbox", _upload_catbox), ("0x0.st", _upload_0x0)):
        try:
            url = fn(path)
            if url:
                logger.info(f"Hosted crop on {name}: {url}")
                return url
        except Exception as e:
            logger.warning(f"{name} upload error: {e}")
    logger.error("All image hosts failed — Lens search will be skipped")
    return None


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m search.image_host <image_path>")
        sys.exit(1)
    print(host_image(sys.argv[1]) or "FAILED")
