"""
search/tineye.py
===============
TinEye reverse-image search — EXACT-image matches (finds the source pages where
the SAME photo appears). Excellent when the input is a photo that's actually
online (e.g., a profile picture) — it points straight at the source URL.

TinEye's API is a paid/commercial service with HMAC-signed requests. This is a
gated integration point: set TINEYE_API_KEY (+ TINEYE_PRIVATE_KEY) in .env and,
depending on your plan, the signing below may need adjusting. Without a key it
skips cleanly.

Returns {"names": [], "leads": [{url, title}, ...]}.
"""

import os
import time
import hmac
import hashlib
import base64
import logging
import requests

logger = logging.getLogger(__name__)

_API = "https://api.tineye.com/rest/search/"


def tineye_available() -> bool:
    return bool(os.getenv("TINEYE_API_KEY"))


def search_tineye(image_path: str) -> dict:
    out = {"names": [], "leads": []}
    api_key = os.getenv("TINEYE_API_KEY")
    private_key = os.getenv("TINEYE_PRIVATE_KEY")
    if not api_key:
        return out
    try:
        with open(image_path, "rb") as f:
            img = f.read()
        params = {"api_key": api_key}
        # If a private key is provided, sign the request (TinEye commercial API).
        if private_key:
            nonce = base64.b64encode(os.urandom(12)).decode()
            date = str(int(time.time()))
            to_sign = f"{private_key}POST{api_key}{nonce}{date}"
            sig = hmac.new(private_key.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
            params.update({"nonce": nonce, "date": date, "api_sig": sig})
        r = requests.post(_API, params=params,
                          files={"image_upload": ("q.jpg", img)}, timeout=25)
        if r.status_code != 200:
            logger.warning(f"TinEye HTTP {r.status_code}: {r.text[:120]}")
            return out
        data = r.json()
        for m in (data.get("results", {}) or {}).get("matches", []) or []:
            for bl in m.get("backlinks", []) or []:
                url = bl.get("backlink") or bl.get("url")
                if url and url.startswith("http"):
                    out["leads"].append({"url": url, "title": bl.get("crawl_date", "")})
        logger.info(f"TinEye: {len(out['leads'])} source pages")
    except Exception as e:
        logger.warning(f"TinEye failed: {e}")
    out["leads"] = out["leads"][:15]
    return out


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    print(json.dumps(search_tineye(sys.argv[1]), indent=2))
