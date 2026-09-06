"""
search/vision.py
===============
Google Cloud Vision — Web Detection. An INDEPENDENT identity signal (separate
from Google Lens): it returns `web_entities` (which very often include the
person's name) plus pages that contain the same image. Great as a second opinion
that agrees/disagrees with Lens → higher-confidence, lower-false-positive voting.

Setup:
  1. In your GCP project, enable the "Cloud Vision API".
  2. Use an API key with Vision access (GOOGLE_VISION_KEY, or reuse GOOGLE_API_KEY).
Free tier: 1,000 images/month.

Returns {"names": [(name, weight), ...], "leads": [{url, title}, ...]}.
"""

import os
import base64
import logging
import requests

logger = logging.getLogger(__name__)

_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Circuit breaker: if Vision returns an auth/billing/quota error once, disable it
# for the rest of the session so we don't fire a failing call on every scan
# (e.g. GOOGLE_VISION_KEY left set but billing not enabled on the project).
_disabled = False


def vision_available() -> bool:
    # Explicit opt-in: requires GOOGLE_VISION_KEY (set it only AFTER enabling the
    # Cloud Vision API), so we never fire failing calls with an unrelated key.
    return bool(os.getenv("GOOGLE_VISION_KEY")) and not _disabled


def search_vision(image_path: str) -> dict:
    global _disabled
    out = {"names": [], "leads": []}
    key = os.getenv("GOOGLE_VISION_KEY")
    if not key or _disabled:
        return out
    try:
        with open(image_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
        body = {"requests": [{"image": {"content": content},
                              "features": [{"type": "WEB_DETECTION", "maxResults": 15}]}]}
        r = requests.post(f"{_ENDPOINT}?key={key}", json=body, timeout=25)
        d = r.json()
        # Auth/billing/quota errors → disable Vision for the rest of the session.
        err_msg = (d.get("error") or {}).get("message")
        resp = (d.get("responses") or [{}])[0]
        err_msg = err_msg or (resp.get("error") or {}).get("message")
        if r.status_code == 403 or err_msg:
            low = (err_msg or "").lower()
            if r.status_code in (401, 403) or any(k in low for k in
                    ("billing", "permission", "disabled", "api key", "quota", "not enabled")):
                _disabled = True
                logger.warning(f"Vision disabled for session ({err_msg or r.status_code})")
            else:
                logger.warning(f"Vision error: {err_msg}")
            return out
        wd = resp.get("webDetection", {}) or {}

        from search.identity import _person_via_spacy, _person_via_regex
        first = True
        for e in wd.get("webEntities", []):
            desc = (e.get("description") or "").strip()
            nm = _person_via_spacy(desc) or _person_via_regex(desc)
            if nm:
                out["names"].append((nm, 2.0 if first else 0.5))
                first = False
        for p in (wd.get("pagesWithMatchingImages") or [])[:15]:
            u = p.get("url")
            if u and u.startswith("http"):
                out["leads"].append({"url": u, "title": p.get("pageTitle", "")})
        logger.info(f"Vision: {len(out['names'])} name entities, {len(out['leads'])} pages")
    except Exception as e:
        logger.warning(f"Vision failed: {e}")
    return out


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    print(json.dumps(search_vision(sys.argv[1]), indent=2))
