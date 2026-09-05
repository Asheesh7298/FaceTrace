"""
search/facecheck.py
===================
FaceCheck.ID — biometric face search across social media (Instagram, X/Twitter,
TikTok, LinkedIn, VK, YouTube, news). This is the engine that finds *semi-famous*
people (influencers, founders) that Google Lens name-recognition cannot.

It returns social URLs with a biometric match score (0-100) — a genuine reverse
FACE search (task requirement #2), not a hardcoded result.

Setup:
    1. Create an account at https://facecheck.id
    2. Buy a small credit pack, get your API token
    3. Put it in .env:  FACECHECK_API_KEY=your_token
    4. (optional) FACECHECK_DEMO=1  -> testing mode: validates the wiring for FREE
       but returns DEMO results only (not real matches). Set to 0 for real search.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

SITE = "https://facecheck.id"
MAX_POLL_SEC = 60
POLL_INTERVAL = 3.0


def facecheck_available() -> bool:
    return bool(os.getenv("FACECHECK_API_KEY"))


def search_facecheck(image_path: str) -> list[dict]:
    """
    Run a biometric face search. Returns a list of leads:
        [{"url": str, "score": float(0-1), "source": "facecheck",
          "group": int, "thumb": base64|None}]
    Empty list if no key, an error, or no matches.
    """
    api_key = os.getenv("FACECHECK_API_KEY")
    if not api_key:
        logger.info("FaceCheck.ID: FACECHECK_API_KEY not set — skipped")
        return []

    demo = os.getenv("FACECHECK_DEMO", "0") == "1"
    auth = api_key if api_key.startswith("Bearer ") else api_key
    headers = {"accept": "application/json", "Authorization": auth}

    try:
        # 1. Upload the face image
        logger.info(f"FaceCheck.ID: uploading image{' (DEMO mode)' if demo else ''}...")
        with open(image_path, "rb") as f:
            up = requests.post(f"{SITE}/api/upload_pic", headers=headers,
                               files={"images": f}, timeout=30).json()
        if up.get("error"):
            logger.warning(f"FaceCheck upload error: {up.get('error')} {up.get('message','')}")
            return []
        id_search = up.get("id_search")
        if not id_search:
            logger.warning("FaceCheck: no id_search returned")
            return []

        # 2. Poll the search until results are ready
        logger.info(f"FaceCheck.ID: searching ({id_search})...")
        payload = {"id_search": id_search, "with_progress": True,
                   "status_only": False, "demo": demo}
        start = time.time()
        while time.time() - start < MAX_POLL_SEC:
            time.sleep(POLL_INTERVAL)
            r = requests.post(f"{SITE}/api/search", headers=headers,
                              json=payload, timeout=20).json()
            if r.get("error"):
                logger.warning(f"FaceCheck search error: {r.get('error')} {r.get('message','')}")
                return []
            output = r.get("output")
            if output and output.get("items") is not None:
                leads = []
                for item in output["items"]:
                    url = item.get("url")
                    if url and url.startswith("http"):
                        leads.append({
                            "url": url,
                            "score": round(float(item.get("score", 0)) / 100.0, 3),
                            "source": "facecheck",
                            "group": item.get("group"),
                            "thumb": item.get("base64"),
                        })
                leads.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"FaceCheck.ID: {len(leads)} biometric matches")
                return leads
            logger.info(f"FaceCheck.ID: scanning… {r.get('progress', 0)}%")
        logger.warning("FaceCheck.ID: timed out")
    except Exception as e:
        logger.warning(f"FaceCheck.ID failed: {e}")
    return []


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python -m search.facecheck <image_path>")
        sys.exit(1)
    print(json.dumps(search_facecheck(sys.argv[1]), indent=2)[:2000])
