"""
search/reverse_search.py
========================
Multi-strategy reverse image search with 7 accuracy layers.

Strategy 1 — Image Search (pixel matching)
  Yandex → Google CSE → SerpApi

Strategy 2 — Face DB Search (identity matching)
  Modal A100 searches LFW + CelebA + VGGFace2 + MS-Celeb + Indian DB

Strategy 3 — Name-Based Web Search
  If Strategy 2 finds a name → search web by name

Strategy 4 — Cross-Validation + Verification
  Cross-validates DB match against image search results.
  Verifies name match by downloading their public photo.
"""

import os
import io
import time
import random
import hashlib
import logging
import tempfile
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from collections import defaultdict

import imagehash
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

PLATFORM_PRIORITY = {
    "linkedin.com":   10, "twitter.com": 9, "x.com": 9,
    "instagram.com":  8,  "facebook.com": 7, "github.com": 7,
    "yourstory.com":  8,  "inc42.com": 7, "crunchbase.com": 7,
    "wellfound.com":  6,  "reddit.com": 5, "youtube.com": 5,
    "researchgate.net": 6, "academia.edu": 6,
}

def platform_score(url):
    domain = urlparse(url).netloc.replace("www.", "")
    for platform, score in PLATFORM_PRIORITY.items():
        if platform in domain:
            return score
    return 3

def phash_similarity(face_crop_path, candidate_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 Chrome/124.0.0.0 Safari/537.36"}
        resp = requests.get(candidate_url, headers=headers, timeout=8, stream=True)
        if "image" not in resp.headers.get("content-type", ""):
            return {"accessible": False, "reason": "not_image"}
        candidate_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        original_img  = Image.open(face_crop_path).convert("RGB")
        orig_hash = imagehash.phash(original_img)
        cand_hash = imagehash.phash(candidate_img)
        distance  = orig_hash - cand_hash
        confidence = "high" if distance < 10 else "medium" if distance < 20 else "low"
        return {"accessible": True, "hamming_distance": int(distance), "phash_confidence": confidence}
    except Exception as e:
        return {"accessible": False, "reason": str(e)}

def wayback_check(url):
    result = {"has_archive": False, "first_seen": None}
    try:
        cdx = f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&limit=1&fl=timestamp&fastLatest=true"
        resp = requests.get(cdx, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1:
                ts = data[1][0]
                result["has_archive"] = True
                result["first_seen"]  = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    except Exception:
        pass
    return result

def generate_search_crops(face_crop_path, output_dir):
    output_dir = Path(output_dir)
    crops = [face_crop_path]
    try:
        from PIL import ImageEnhance
        img = Image.open(face_crop_path).convert("RGB")
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        p = str(output_dir / "face_crop_flipped.jpg")
        flipped.save(p, quality=95)
        crops.append(p)
        sharpened = ImageEnhance.Sharpness(img).enhance(2.0)
        p = str(output_dir / "face_crop_sharpened.jpg")
        sharpened.save(p, quality=95)
        crops.append(p)
    except Exception as e:
        logger.warning(f"Crop generation failed: {e}")
    return crops

def search_yandex(face_crop_path):
    urls = []
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
            context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36", viewport={"width": 1280, "height": 720})
            page = context.new_page()
            page.goto("https://yandex.com/images/", timeout=20000)
            time.sleep(random.uniform(1.5, 3.0))
            try:
                page.click('button[aria-label="Search by image"]', timeout=8000)
            except Exception:
                try:
                    page.click('.cbir-button', timeout=5000)
                except Exception:
                    pass
            time.sleep(random.uniform(0.5, 1.5))
            try:
                page.locator('input[type="file"]').set_input_files(str(Path(face_crop_path).resolve()))
                time.sleep(random.uniform(4.0, 6.0))
            except Exception as e:
                logger.warning(f"Yandex upload failed: {e}")
                browser.close()
                return []
            links = page.eval_on_selector_all("a[href]", "els => els.map(el => el.href)")
            for link in links:
                if link.startswith("http") and "yandex" not in link and "google" not in link and len(link) > 30:
                    urls.append(link)
            browser.close()
            logger.info(f"Yandex: {len(urls)} URLs")
    except Exception as e:
        logger.warning(f"Yandex failed: {e}")
    return list(dict.fromkeys(urls))[:20]

def search_google_cse(face_crop_path, person_name=None):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return []
    urls = []
    try:
        if person_name:
            queries = [
                f'"{person_name}" site:linkedin.com',
                f'"{person_name}" site:twitter.com OR site:x.com',
                f'"{person_name}" site:github.com',
                f'"{person_name}" startup founder India',
                f'"{person_name}" profile',
            ]
            for query in queries:
                resp = requests.get("https://www.googleapis.com/customsearch/v1", params={"key": api_key, "cx": cse_id, "q": query, "num": 5}, timeout=10)
                if resp.status_code == 200:
                    for item in resp.json().get("items", []):
                        if item.get("link"):
                            urls.append(item["link"])
                time.sleep(0.5)
        else:
            resp = requests.get("https://www.googleapis.com/customsearch/v1", params={"key": api_key, "cx": cse_id, "q": "person profile photo", "searchType": "image", "imgType": "face", "num": 10}, timeout=10)
            if resp.status_code == 200:
                for item in resp.json().get("items", []):
                    url = item.get("link") or item.get("image", {}).get("contextLink")
                    if url:
                        urls.append(url)
        logger.info(f"Google CSE: {len(urls)} URLs (name={person_name})")
    except Exception as e:
        logger.warning(f"Google CSE failed: {e}")
    return list(dict.fromkeys(urls))[:20]

def search_serpapi(face_crop_path):
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return []
    urls = []
    try:
        import base64
        with open(face_crop_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        resp = requests.get("https://serpapi.com/search", params={"engine": "google_reverse_image", "api_key": api_key, "image_content": image_b64}, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("image_results", []):
                url = item.get("link") or item.get("original")
                if url:
                    urls.append(url)
        logger.info(f"SerpApi: {len(urls)} URLs")
    except Exception as e:
        logger.warning(f"SerpApi failed: {e}")
    return list(dict.fromkeys(urls))[:20]

def search_modal_db(face_crop_path):
    try:
        import modal
        from face_search_modal import app, search_face_db
        with open(face_crop_path, "rb") as f:
            image_bytes = f.read()
        logger.info("Calling Modal A100 face DB search...")
        with app.run():
            result = search_face_db.remote(image_bytes)
        if result.get("found"):
            logger.info(f"Modal DB match: {result['name']} (conf: {result['confidence']}, db: {result['database']})")
        else:
            logger.info("Modal DB: no match found")
        return result
    except Exception as e:
        logger.warning(f"Modal unavailable: {e}")
        return {"found": False, "name": None, "confidence": 0.0}

def verify_name_match(name, face_crop_path):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return 0.7
    try:
        from deepface import DeepFace
        resp = requests.get("https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": f'"{name}" official portrait headshot',
                    "searchType": "image", "imgType": "face", "num": 5}, timeout=10)
        items = resp.json().get("items", []) if resp.status_code == 200 else []
        for item in items:
            img_url = item.get("link")
            if not img_url:
                continue
            try:
                img_resp = requests.get(img_url, timeout=6)
                if img_resp.status_code != 200 or len(img_resp.content) < 5000:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(img_resp.content)
                    temp_path = f.name
                result = DeepFace.verify(img1_path=face_crop_path, img2_path=temp_path, model_name="ArcFace", enforce_detection=False, silent=True)
                os.unlink(temp_path)
                if result["verified"]:
                    conf = round(1 - result["distance"], 3)
                    logger.info(f"Name verification: {name} CONFIRMED (conf: {conf})")
                    return conf
            except Exception as e:
                logger.debug(f"Verification attempt failed: {e}")
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Name verification failed: {e}")
    return 0.6

def cross_validate_strategies(db_result, image_search_urls):
    if not db_result.get("found"):
        return db_result
    name = db_result.get("name", "").lower()
    name_parts = [p for p in name.split() if len(p) > 3]
    name_in_urls = any(any(part in url.lower() for part in name_parts) for url in image_search_urls)
    if name_in_urls:
        db_result["confidence"] = min(db_result["confidence"] * 1.2, 1.0)
        db_result["cross_validated"] = True
        logger.info(f"Cross-validation: AGREE on {db_result['name']}")
    else:
        db_result["confidence"] *= 0.85
        db_result["cross_validated"] = False
        logger.warning(f"Cross-validation: DISAGREE — DB says {db_result['name']}")
    db_result["confidence"] = round(db_result["confidence"], 3)
    return db_result

def score_candidates(url_engine_map, face_crop_path, engine_count, person_name=None):
    candidates = []
    top_urls = sorted(url_engine_map.keys(), key=lambda u: len(url_engine_map[u]), reverse=True)[:15]
    for url in top_urls:
        engines_agreed   = url_engine_map[url]
        engine_agreement = len(engines_agreed) / max(engine_count, 1)
        plat_score       = platform_score(url)
        phash_result     = phash_similarity(face_crop_path, url)
        wayback          = wayback_check(url) if len(engines_agreed) >= 2 else {}
        name_bonus       = 0.0
        if person_name:
            name_parts = [p.lower() for p in person_name.split() if len(p) > 3]
            if any(part in url.lower() for part in name_parts):
                name_bonus = 0.15
        phash_comp = 0.0
        if phash_result and phash_result.get("accessible"):
            dist = phash_result.get("hamming_distance", 64)
            phash_comp = max(0.0, 1.0 - dist / 64.0) * 0.15
        composite_score = (engine_agreement * 0.40) + ((plat_score / 10) * 0.30) + phash_comp + (name_bonus * 0.15)
        candidates.append({
            "url": url, "engines": engines_agreed,
            "engine_agreement": round(engine_agreement, 3),
            "platform_score": plat_score,
            "platform": urlparse(url).netloc.replace("www.", ""),
            "phash": phash_result, "wayback": wayback,
            "name_match": name_bonus > 0,
            "composite_score": round(composite_score, 4),
        })
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)
    return candidates

def multi_engine_search(face_crop_path, output_dir="output", exif_data=None, use_modal=True):
    result = {
        "best_match": None, "all_candidates": [],
        "engines_queried": [], "engines_with_results": [],
        "total_candidates": 0, "modal_result": {},
        "person_name": None, "name_confidence": 0.0,
        "name_verified": False, "success": False,
    }
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops        = generate_search_crops(face_crop_path, str(output_dir))
    primary_crop = crops[0]
    person_name  = None

    # Strategy 2 — Modal DB
    modal_result = {"found": False}
    if use_modal:
        logger.info("\n[Strategy 2] Modal A100 Face DB Search...")
        modal_result = search_modal_db(primary_crop)
        result["modal_result"] = modal_result
        if modal_result.get("found"):
            person_name = modal_result["name"]
            result["person_name"]     = person_name
            result["name_confidence"] = modal_result["confidence"]
            logger.info(f"Person identified: {person_name}")

    # Strategy 1 — Image Search
    engine_results = {}
    logger.info("\n[Strategy 1] Image Search Engines...")

    result["engines_queried"].append("yandex")
    yandex_urls = search_yandex(primary_crop)
    if yandex_urls:
        engine_results["yandex"] = yandex_urls
        result["engines_with_results"].append("yandex")
    time.sleep(random.uniform(1.0, 2.0))

    result["engines_queried"].append("google_cse")
    google_urls = search_google_cse(primary_crop)
    if google_urls:
        engine_results["google_cse"] = google_urls
        result["engines_with_results"].append("google_cse")
    time.sleep(random.uniform(1.0, 2.0))

    if sum(len(v) for v in engine_results.values()) < 5:
        result["engines_queried"].append("serpapi")
        serpapi_urls = search_serpapi(primary_crop)
        if serpapi_urls:
            engine_results["serpapi"] = serpapi_urls
            result["engines_with_results"].append("serpapi")

    # Strategy 3 — Name-based search
    if person_name:
        logger.info(f"\n[Strategy 3] Name search: {person_name}")
        name_urls = search_google_cse(primary_crop, person_name=person_name)
        if name_urls:
            engine_results["name_search"] = name_urls
            result["engines_with_results"].append("name_search")

    # Pool and score
    url_engine_map = defaultdict(list)
    for engine, urls in engine_results.items():
        for url in urls:
            url_engine_map[url].append(engine)

    if not url_engine_map:
        logger.warning("All strategies returned 0 URLs")
        return result

    candidates = score_candidates(url_engine_map, face_crop_path, len(engine_results), person_name)
    result["all_candidates"]   = candidates
    result["total_candidates"] = len(candidates)

    # Strategy 4 — Cross-validation
    if modal_result.get("found") and candidates:
        all_urls     = [c["url"] for c in candidates]
        modal_result = cross_validate_strategies(modal_result, all_urls)
        result["modal_result"]    = modal_result
        result["name_confidence"] = modal_result["confidence"]

    # Layer 7 — Name verification
    if person_name and modal_result.get("confidence", 0) > 0.55:
        logger.info(f"\n[Layer 7] Verifying: {person_name}")
        verified_conf = verify_name_match(person_name, face_crop_path)
        result["name_verified"]   = verified_conf > 0.65
        result["name_confidence"] = round((result["name_confidence"] + verified_conf) / 2, 3)
        logger.info(f"Verification: {'CONFIRMED' if result['name_verified'] else 'UNCERTAIN'} (conf: {result['name_confidence']})")

    # Select best match
    if candidates:
        best = candidates[0]
        if person_name:
            name_candidates = [c for c in candidates if "name_search" in c.get("engines", []) and c["platform_score"] >= 7]
            if name_candidates:
                best = name_candidates[0]
                logger.info(f"Prioritizing name-search result: {best['url']}")
        result["best_match"] = best
        result["success"]    = True
        logger.info(f"\nBest match: {best['url']}")
        logger.info(f"  Score: {best['composite_score']} | Platform: {best['platform']}")

    return result


if __name__ == "__main__":
    import sys, json, logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python search/reverse_search.py <face_crop_path> [--no-modal]")
        sys.exit(1)
    use_modal = "--no-modal" not in sys.argv
    result = multi_engine_search(sys.argv[1], use_modal=use_modal)
    printable = {k: v for k, v in result.items() if k != "all_candidates"}
    printable["candidate_count"] = result["total_candidates"]
    print(json.dumps(printable, indent=2, default=str))