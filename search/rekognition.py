"""
search/rekognition.py
====================
AWS Rekognition — RecognizeCelebrities. A dedicated celebrity face recognizer:
give it the face, it returns the celebrity's NAME + confidence, plus URLs
(usually IMDb/Wikipedia). A strong, independent identity signal for famous people.

Setup:
  pip install boto3
  Create a free-tier AWS account, then set in .env:
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_REGION=us-east-1        (optional, defaults to us-east-1)
Free tier: 5,000 images/month for the first 12 months.

Returns {"names": [(name, weight), ...], "leads": [{url, title}, ...]}.
"""

import os
import logging

logger = logging.getLogger(__name__)


def rekognition_available() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def search_rekognition(image_path: str, min_confidence: float = 80.0) -> dict:
    out = {"names": [], "leads": []}
    if not rekognition_available():
        return out
    try:
        import boto3
        client = boto3.client("rekognition",
                              region_name=os.getenv("AWS_REGION", "us-east-1"))
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        resp = client.recognize_celebrities(Image={"Bytes": img_bytes})
        for c in resp.get("CelebrityFaces", []):
            conf = float(c.get("MatchConfidence", 0))
            if conf < min_confidence:
                continue
            name = c.get("Name")
            if not name:
                continue
            out["names"].append((name, 1.5 + conf / 100.0))     # 1.5–2.5
            for u in c.get("Urls", []) or []:
                url = u if u.startswith("http") else "https://" + u
                out["leads"].append({"url": url, "title": name})
        logger.info(f"Rekognition: {len(out['names'])} celebrity match(es)")
    except Exception as e:
        logger.warning(f"Rekognition failed: {e}")
    return out


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    print(json.dumps(search_rekognition(sys.argv[1]), indent=2))
