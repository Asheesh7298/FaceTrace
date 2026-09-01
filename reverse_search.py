"""
search/reverse_search.py
Multi-engine reverse image search with result ranking.

Engines (in priority order):
  1. Yandex    — best face accuracy, free, Playwright scraper
  2. Bing      — official API, 1000 free/month
  3. SerpApi   — Google Reverse Image, 100 free/month (last resort)

Ranking:
  - Engine agreement score  (URL found by 2+ engines = ranked higher)
  - pHash similarity        (Hamming distance on downloadable images)
  - Platform priority       (linkedin > twitter > instagram > facebook > other)

Output:
  Best candidate URL + full ranked list + confidence signals
"""

import os
import time
import random
import hashlib
import logging
import requests
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from collections import defaultdict

import imagehash
from PIL import Image

logger = logging.getLogger(__name__)

# ─── Platform Priority ────────────────────────────────────────────────────────

PLATFORM_PRIORITY = {
    "linkedin.com":   10,
    "twitter.com":    9,
    "x.com":          9,
    "instagram.com":  8,
    "facebook.com":   7,
    "github.com":     7,
    "reddit.com":     6,
    "youtube.com":    5,
}

def platform_score(url: str) -> int:
    domain = urlparse(url).netloc.replace("www.", "")
    for platform, score in PLATFORM_PRIORITY.items():
        if platform in domain:
            return score
    return 3  # generic web


# ─── pHash Similarity ─────────────────────────────────────────────────────────

def phash_similarity(face_crop_path: str, candidate_url: str) -> Optional[dict]:
    """
    Downloads candidate image and computes perceptual hash similarity.
    Returns None if image not downloadable (auth wall, HTML page, etc.)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(candidate_url, headers=headers, timeout=8, stream=True)

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return {"accessible": False, "reason": "not_an_image_url"}

        img_bytes = resp.content
        candidate_img = Image.open(__import__("io").BytesIO(img_bytes)).convert("RGB")
        original_img  = Image.open(face_crop_path).convert("RGB")

        original_hash  = imagehash.phash(original_img)
        candidate_hash = imagehash.phash(candidate_img)
        distance = original_hash - candidate_hash

        if distance < 10:
            confidence = "high"
        elif distance < 20:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "accessible": True,
            "hamming_distance": int(distance),
            "phash_confidence": confidence,
        }

    except Exception as e:
        return {"accessible": False, "reason": str(e)}


# ─── Engine 1: Yandex (Playwright scraper) ────────────────────────────────────

def search_yandex(face_crop_path: str) -> list[str]:
    """
    Uploads image to Yandex reverse image search via Playwright.
    Returns list of result URLs.
    """
    urls = []
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                ],
            )

            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()

            logger.info("Yandex: navigating to images.yandex.com...")
            page.goto("https://yandex.com/images/", timeout=20000)
            time.sleep(random.uniform(1.5, 3.0))

            # Click the camera icon (search by image)
            page.click('button[aria-label="Search by image"]', timeout=8000)
            time.sleep(random.uniform(0.5, 1.5))

            # Upload the face crop
            file_input = page.locator('input[type="file"]')
            file_input.set_input_files(str(Path(face_crop_path).resolve()))
            logger.info("Yandex: image uploaded, waiting for results...")
            time.sleep(random.uniform(4.0, 6.0))

            # Parse result links
            links = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(el => el.href)"
            )

            for link in links:
                if (
                    link.startswith("http")
                    and "yandex" not in link
                    and "google" not in link
                    and len(link) > 30
                ):
                    urls.append(link)

            browser.close()
            logger.info(f"Yandex: found {len(urls)} raw URLs")

    except Exception as e:
        logger.warning(f"Yandex search failed: {e}")

    return list(dict.fromkeys(urls))[:20]  # deduplicate, cap at 20


# ─── Engine 2: Bing Visual Search ────────────────────────────────────────────

def search_bing(face_crop_path: str) -> list[str]:
    """
    Bing Visual Search API — uploads image bytes directly.
    Returns list of result URLs.
    """
    api_key = os.getenv("BING_SEARCH_KEY")
    if not api_key:
        logger.warning("BING_SEARCH_KEY not set — skipping Bing")
        return []

    urls = []
    try:
        with open(face_crop_path, "rb") as f:
            image_data = f.read()

        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "multipart/form-data",
        }

        response = requests.post(
            "https://api.bing.microsoft.com/v7.0/images/visualsearch",
            headers={"Ocp-Apim-Subscription-Key": api_key},
            files={"image": ("face_crop.jpg", image_data, "image/jpeg")},
            timeout=15,
        )

        if response.status_code != 200:
            logger.warning(f"Bing API error: {response.status_code} — {response.text[:200]}")
            return []

        data = response.json()

        # Extract URLs from all action types
        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                action_type = action.get("actionType", "")

                # Pages containing the image
                if action_type == "PagesIncluding":
                    for item in action.get("data", {}).get("value", []):
                        url = item.get("hostPageUrl") or item.get("contentUrl")
                        if url:
                            urls.append(url)

                # Visual search results
                elif action_type == "VisualSearch":
                    for item in action.get("data", {}).get("value", []):
                        url = item.get("hostPageUrl") or item.get("contentUrl")
                        if url:
                            urls.append(url)

        logger.info(f"Bing: found {len(urls)} raw URLs")

    except Exception as e:
        logger.warning(f"Bing search failed: {e}")

    return list(dict.fromkeys(urls))[:20]


# ─── Engine 3: SerpApi (Google Reverse Image) ─────────────────────────────────

def search_serpapi(face_crop_path: str) -> list[str]:
    """
    SerpApi Google Reverse Image search.
    Uploads image, returns list of result URLs.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        logger.warning("SERPAPI_KEY not set — skipping SerpApi")
        return []

    urls = []
    try:
        import base64

        with open(face_crop_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        # SerpApi accepts base64 image for reverse image search
        params = {
            "engine": "google_reverse_image",
            "api_key": api_key,
            "image_content": image_b64,
        }

        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=20,
        )

        if response.status_code != 200:
            logger.warning(f"SerpApi error: {response.status_code}")
            return []

        data = response.json()

        # Pages with matching images
        for item in data.get("image_results", []):
            url = item.get("link") or item.get("original")
            if url:
                urls.append(url)

        # Inline images with source links
        for item in data.get("inline_images", []):
            url = item.get("link")
            if url:
                urls.append(url)

        logger.info(f"SerpApi: found {len(urls)} raw URLs")

    except Exception as e:
        logger.warning(f"SerpApi search failed: {e}")

    return list(dict.fromkeys(urls))[:20]


# ─── Multi-Crop Strategy ─────────────────────────────────────────────────────

def generate_search_crops(face_crop_path: str, output_dir: str) -> list[str]:
    """
    Generates 3 crop variants for broader engine coverage:
      - tight  : face only (original crop)
      - flipped: horizontal mirror (catches different angle matches)
      - sharpened: contrast-enhanced (helps on blurry inputs)
    """
    output_dir = Path(output_dir)
    crops = [face_crop_path]  # original always included

    try:
        img = Image.open(face_crop_path).convert("RGB")

        # Flipped
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_path = str(output_dir / "face_crop_flipped.jpg")
        flipped.save(flipped_path, quality=95)
        crops.append(flipped_path)

        # Sharpened
        from PIL import ImageEnhance, ImageFilter
        sharpened = img.filter(ImageFilter.SHARPEN)
        sharpened = ImageEnhance.Contrast(sharpened).enhance(1.3)
        sharpened_path = str(output_dir / "face_crop_sharpened.jpg")
        sharpened.save(sharpened_path, quality=95)
        crops.append(sharpened_path)

    except Exception as e:
        logger.warning(f"Crop variant generation failed: {e}")

    return crops


# ─── Wayback Machine Cross-Reference ─────────────────────────────────────────

def wayback_check(url: str) -> dict:
    """
    Checks if a URL has Wayback Machine snapshots.
    Returns first-seen date and snapshot count.
    """
    result = {"has_archive": False, "first_seen": None, "snapshot_count": 0}
    try:
        cdx_url = (
            f"https://web.archive.org/cdx/search/cdx"
            f"?url={url}&output=json&limit=1&fl=timestamp&fastLatest=true"
        )
        resp = requests.get(cdx_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1:  # first row is header
                ts = data[1][0]  # e.g. "20231102142200"
                result["has_archive"] = True
                result["first_seen"] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

        # Snapshot count (separate query)
        count_url = f"https://web.archive.org/cdx/search/cdx?url={url}&output=json&fl=timestamp&limit=100"
        count_resp = requests.get(count_url, timeout=5)
        if count_resp.status_code == 200:
            count_data = count_resp.json()
            result["snapshot_count"] = max(0, len(count_data) - 1)

    except Exception as e:
        logger.debug(f"Wayback check failed for {url}: {e}")

    return result


# ─── Main Search Function ─────────────────────────────────────────────────────

def multi_engine_search(
    face_crop_path: str,
    output_dir: str = "output",
    exif_data: Optional[dict] = None,
) -> dict:
    """
    Runs multi-engine reverse image search and returns ranked results.

    Returns:
      {
        "best_match": { url, platform, engine_agreement, phash, wayback, score },
        "all_candidates": [ ... ],
        "engines_queried": [...],
        "engines_with_results": [...],
        "total_candidates": int,
        "success": bool
      }
    """
    result = {
        "best_match": None,
        "all_candidates": [],
        "engines_queried": [],
        "engines_with_results": [],
        "total_candidates": 0,
        "success": False,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate search crop variants
    crops = generate_search_crops(face_crop_path, str(output_dir))
    primary_crop = crops[0]

    # ── Run engines ───────────────────────────────────────────────────────────
    engine_results: dict[str, list[str]] = {}

    logger.info("=" * 50)
    logger.info("Starting multi-engine reverse image search")
    logger.info("=" * 50)

    # Yandex (primary — best for faces)
    logger.info("\n[Engine 1/3] Yandex...")
    result["engines_queried"].append("yandex")
    yandex_urls = search_yandex(primary_crop)
    if yandex_urls:
        engine_results["yandex"] = yandex_urls
        result["engines_with_results"].append("yandex")
        logger.info(f"Yandex: {len(yandex_urls)} results")
    else:
        logger.warning("Yandex: 0 results")

    time.sleep(random.uniform(1.0, 2.5))  # jitter between engines

    # Bing (secondary)
    logger.info("\n[Engine 2/3] Bing Visual Search...")
    result["engines_queried"].append("bing")
    bing_urls = search_bing(primary_crop)
    if bing_urls:
        engine_results["bing"] = bing_urls
        result["engines_with_results"].append("bing")
        logger.info(f"Bing: {len(bing_urls)} results")
    else:
        logger.warning("Bing: 0 results")

    time.sleep(random.uniform(1.0, 2.5))

    # SerpApi (last resort — preserves free tier)
    all_results_so_far = sum(len(v) for v in engine_results.values())
    if all_results_so_far < 3:
        logger.info("\n[Engine 3/3] SerpApi (Google Reverse Image)...")
        result["engines_queried"].append("serpapi")
        serpapi_urls = search_serpapi(primary_crop)
        if serpapi_urls:
            engine_results["serpapi"] = serpapi_urls
            result["engines_with_results"].append("serpapi")
            logger.info(f"SerpApi: {len(serpapi_urls)} results")
        else:
            logger.warning("SerpApi: 0 results")
    else:
        logger.info("\n[Engine 3/3] SerpApi skipped (sufficient results from other engines)")

    # ── Pool and score all URLs ───────────────────────────────────────────────
    url_engine_map: dict[str, list[str]] = defaultdict(list)
    for engine, urls in engine_results.items():
        for url in urls:
            url_engine_map[url].append(engine)

    if not url_engine_map:
        logger.warning("All engines returned 0 results")
        result["success"] = False
        return result

    # ── Score each candidate ──────────────────────────────────────────────────
    candidates = []
    top_urls = sorted(
        url_engine_map.keys(),
        key=lambda u: len(url_engine_map[u]),
        reverse=True
    )[:15]  # Only score top 15 by engine agreement

    logger.info(f"\nScoring top {len(top_urls)} candidates...")

    for url in top_urls:
        engines_agreed = url_engine_map[url]
        engine_agreement = len(engines_agreed) / max(len(engine_results), 1)
        plat_score = platform_score(url)

        # pHash check (fast — runs on all candidates)
        phash_result = phash_similarity(face_crop_path, url)

        # Wayback Machine (only for top candidates by engine agreement)
        wayback = {}
        if len(engines_agreed) >= 2:
            wayback = wayback_check(url)

        # Composite score
        agreement_component = engine_agreement * 0.45
        platform_component  = (plat_score / 10) * 0.30
        phash_component     = 0.0

        if phash_result and phash_result.get("accessible"):
            dist = phash_result.get("hamming_distance", 64)
            # Hamming 0 = identical → 1.0; Hamming 64 = completely different → 0.0
            phash_sim = max(0.0, 1.0 - dist / 64.0)
            phash_component = phash_sim * 0.25

        composite_score = agreement_component + platform_component + phash_component

        candidate = {
            "url": url,
            "engines": engines_agreed,
            "engine_agreement": round(engine_agreement, 3),
            "platform_score": plat_score,
            "platform": urlparse(url).netloc.replace("www.", ""),
            "phash": phash_result,
            "wayback": wayback,
            "composite_score": round(composite_score, 4),
        }
        candidates.append(candidate)

    # Sort by composite score
    candidates.sort(key=lambda c: c["composite_score"], reverse=True)
    result["all_candidates"] = candidates
    result["total_candidates"] = len(candidates)

    if candidates:
        result["best_match"] = candidates[0]
        result["success"] = True
        logger.info(f"\nBest match: {candidates[0]['url']}")
        logger.info(f"  Engines agreed: {candidates[0]['engines']}")
        logger.info(f"  Composite score: {candidates[0]['composite_score']}")
        logger.info(f"  Platform: {candidates[0]['platform']}")

    return result


if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python search/reverse_search.py <face_crop_path>")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()

    result = multi_engine_search(sys.argv[1])
    # Pretty print without large nested objects
    print(json.dumps(result, indent=2, default=str))
