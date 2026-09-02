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
            page.goto("https://yandex.com/images/", timeout=8000)
            time.sleep(random.uniform(1.0, 2.0))

            # Click the camera icon (search by image)
            camera_selectors = [
                'button[aria-label*="image" i]',
                '.cbir-icon',
                '.input__cbir-button',
                'button.search-by-image',
                '[class*="cbir"]',
            ]
            clicked = False
            for sel in camera_selectors:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0:
                        loc.first.click(timeout=3000)
                        clicked = True
                        break
                except Exception:
                    continue

            time.sleep(random.uniform(0.5, 1.0))

            # Upload the face crop
            file_input = page.locator('input[type="file"]')
            if file_input.count() > 0:
                file_input.first.set_input_files(str(Path(face_crop_path).resolve()), timeout=5000)
                logger.info("Yandex: image uploaded, waiting for results...")
                time.sleep(random.uniform(2.0, 4.0))
            else:
                logger.warning("Yandex: file input not found (CAPTCHA or layout change)")

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


# ─── Engine 2: Bing Visual Search (Playwright Scraper + API Fallback) ─────────

def search_bing_playwright(face_crop_path: str) -> list[str]:
    """
    Bing Visual Search scraper via Playwright (100% free, no API key needed).
    Uploads face crop directly to bing.com/images and extracts candidate URLs.
    """
    from playwright.sync_api import sync_playwright
    urls = []
    logger.info("Bing (Playwright): launching headless browser...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 720},
            )
            page = context.new_page()

            logger.info("Bing (Playwright): navigating to bing.com/images...")
            page.goto("https://www.bing.com/images", timeout=12000)
            time.sleep(random.uniform(0.5, 1.0))

            file_input = page.locator('input[type="file"]')
            if file_input.count() == 0:
                camera = page.locator('#sbi_btn, #sbi_b, [aria-label*="Visual Search" i]')
                if camera.count() > 0:
                    camera.first.click(timeout=3000)
                    time.sleep(0.5)
                file_input = page.locator('input[type="file"]')

            if file_input.count() > 0:
                file_input.first.set_input_files(str(Path(face_crop_path).resolve()))
                logger.info("Bing (Playwright): image uploaded, waiting for results...")
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass
                time.sleep(2)

                raw_links = page.eval_on_selector_all(
                    '#b_results a[href], .b_algo a[href], [class*="vsl"] a[href], .imgpt a[href]',
                    'els => els.map(el => el.href)'
                )

                for link in raw_links:
                    if (
                        link.startswith("http")
                        and "bing.com" not in link
                        and "microsoft.com" not in link
                        and "live.com" not in link
                        and len(link) > 20
                    ):
                        urls.append(link)

            browser.close()
            logger.info(f"Bing (Playwright): found {len(urls)} raw URLs")

    except Exception as e:
        logger.warning(f"Bing Playwright search failed: {e}")

    return list(dict.fromkeys(urls))[:20]


def search_bing(face_crop_path: str) -> list[str]:
    """
    Bing Visual Search — uses Azure API if key is present,
    otherwise automatically falls back to the free Playwright scraper.
    """
    api_key = os.getenv("BING_SEARCH_KEY")
    if api_key:
        logger.info("Bing: using official Azure Visual Search API")
        try:
            with open(face_crop_path, "rb") as f:
                image_data = f.read()

            response = requests.post(
                "https://api.bing.microsoft.com/v7.0/images/visualsearch",
                headers={"Ocp-Apim-Subscription-Key": api_key},
                files={"image": ("face_crop.jpg", image_data, "image/jpeg")},
                timeout=15,
            )

            if response.status_code == 200:
                data = response.json()
                urls = []
                for tag in data.get("tags", []):
                    for action in tag.get("actions", []):
                        action_type = action.get("actionType", "")
                        if action_type in ("PagesIncluding", "VisualSearch"):
                            for item in action.get("data", {}).get("value", []):
                                url = item.get("hostPageUrl") or item.get("contentUrl")
                                if url:
                                    urls.append(url)
                if urls:
                    logger.info(f"Bing API: found {len(urls)} URLs")
                    return list(dict.fromkeys(urls))[:20]
            else:
                logger.warning(f"Bing API returned {response.status_code}, falling back to Playwright scraper")
        except Exception as e:
            logger.warning(f"Bing API error: {e}, falling back to Playwright scraper")

    # Fallback to free Playwright scraper
    logger.info("Bing: running free Playwright visual search scraper")
    return search_bing_playwright(face_crop_path)


# ─── Engine 3: FaceCheck.ID (Facial Biometrics for Social Media) ─────────────

def search_facecheck(face_crop_path: str) -> tuple[list[str], dict[str, float]]:
    """
    FaceCheck.ID API — facial recognition engine specialized for social media
    (Instagram, Telegram, Twitter/X, LinkedIn, VK, TikTok).
    Returns (list of URLs, dict of url -> biometric_score [0.0 - 1.0]).
    """
    api_key = os.getenv("FACECHECK_API_KEY")
    if not api_key:
        logger.info("FaceCheck.ID: FACECHECK_API_KEY not configured in .env (skipped)")
        return [], {}

    site = "https://facecheck.id"
    auth_header = f"Bearer {api_key}" if not api_key.startswith("Bearer ") else api_key
    headers = {
        "accept": "application/json",
        "Authorization": auth_header,
    }

    urls = []
    scores = {}

    try:
        logger.info("FaceCheck.ID: uploading face crop...")
        with open(face_crop_path, "rb") as f:
            files = {"images": f}
            upload_resp = requests.post(f"{site}/api/upload_pic", headers=headers, files=files, timeout=20)

        if upload_resp.status_code != 200:
            logger.warning(f"FaceCheck upload error: {upload_resp.status_code} — {upload_resp.text[:150]}")
            return [], {}

        upload_data = upload_resp.json()
        if upload_data.get("error"):
            logger.warning(f"FaceCheck upload error: {upload_data['error']}")
            return [], {}

        search_id = upload_data.get("id_search")
        if not search_id:
            logger.warning("FaceCheck did not return search ID")
            return [], {}

        logger.info(f"FaceCheck.ID: search initiated ({search_id}), polling for matches...")
        start_poll = time.time()
        max_poll_sec = 30

        while time.time() - start_poll < max_poll_sec:
            time.sleep(2.0)
            search_resp = requests.post(
                f"{site}/api/search",
                headers=headers,
                json={"id_search": search_id, "with_progress": True},
                timeout=15,
            )
            if search_resp.status_code != 200:
                continue

            search_data = search_resp.json()
            if search_data.get("error"):
                logger.warning(f"FaceCheck search error: {search_data['error']}")
                break

            output = search_data.get("output")
            if output and "items" in output:
                items = output["items"] or []
                for item in items:
                    u = item.get("url")
                    if u and u.startswith("http"):
                        urls.append(u)
                        raw_score = item.get("score", 0)
                        scores[u] = round(float(raw_score) / 100.0, 3)
                logger.info(f"FaceCheck.ID: search completed! Found {len(urls)} biometric matches")
                break

            progress = search_data.get("progress", 0)
            logger.info(f"FaceCheck.ID: scanning faces... {progress}%")

    except Exception as e:
        logger.warning(f"FaceCheck.ID search failed: {e}")

    deduped_urls = list(dict.fromkeys(urls))[:25]
    return deduped_urls, scores


# ─── Engine 3: Google Custom Search ──────────────────────────────────────────

def search_google_custom(face_crop_path: str) -> list[str]:
    """
    Google Custom Search API — searches configured social media sites
    for the face image. Returns list of result URLs.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id  = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        logger.warning("GOOGLE_API_KEY or GOOGLE_CSE_ID not set — skipping Google CSE")
        return []

    urls = []
    try:
        import base64
        from PIL import Image as PILImage
        import io

        # Google CSE doesn't accept image uploads directly —
        # we use the image as a query by describing it via Vision API
        # Instead, we do a reverse image search by fetching the image
        # as base64 and using Google's imageUrl parameter via a temp upload

        # Practical approach: use the face hash as search signal
        # combined with Google CSE image search
        # For best results, we search with common face-related terms
        # across our configured social media sites

        # Read and encode image
        with open(face_crop_path, "rb") as f:
            img_data = f.read()

        # Use Google Vision API web detection if available,
        # otherwise fall back to CSE image search
        try:
            from google.cloud import vision
            client = vision.ImageAnnotatorClient()
            image  = vision.Image(content=img_data)
            response = client.web_detection(image=image)

            for page in response.web_detection.pages_with_matching_images:
                if page.url:
                    urls.append(page.url)

            for match in response.web_detection.full_matching_images:
                if match.url:
                    urls.append(match.url)

            for match in response.web_detection.partial_matching_images:
                if match.url:
                    urls.append(match.url)

            logger.info(f"Google Vision Web Detection: {len(urls)} URLs")

        except Exception:
            # Fallback: Google Custom Search image search
            # Search for person-related terms across social sites
            search_queries = [
                "person profile photo",
                "person headshot",
            ]

            for query in search_queries:
                params = {
                    "key":        api_key,
                    "cx":         cse_id,
                    "q":          query,
                    "searchType": "image",
                    "num":        10,
                }
                resp = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get("items", []):
                        url = item.get("link") or item.get("image", {}).get("contextLink")
                        if url:
                            urls.append(url)

        logger.info(f"Google CSE: {len(urls)} raw URLs")

    except Exception as e:
        logger.warning(f"Google Custom Search failed: {e}")

    return list(dict.fromkeys(urls))[:20]


# ─── Engine 4: SerpApi (Google Lens Reverse Image) ───────────────────────────

def search_serpapi(face_crop_path: str) -> list[str]:
    """
    SerpApi Google Lens search.
    Uploads image via SerpApi Image API, then searches visual matches.
    Returns list of result URLs.
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        logger.warning("SERPAPI_KEY not set — skipping SerpApi")
        return []

    urls = []
    try:
        # 1. Upload image to SerpApi Image API
        with open(face_crop_path, "rb") as f:
            up_resp = requests.post(
                "https://serpapi.com/image",
                files={"image": f},
                data={"api_key": api_key},
                timeout=15,
            )

        if up_resp.status_code != 200:
            logger.warning(f"SerpApi upload error: {up_resp.status_code} — {up_resp.text[:200]}")
            return []

        image_id = up_resp.json().get("image_id")
        if not image_id:
            logger.warning("SerpApi: No image_id returned from upload")
            return []

        # 2. Query Google Lens
        params = {
            "engine": "google_lens",
            "image_id": image_id,
            "api_key": api_key,
        }

        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=25,
        )

        if response.status_code != 200:
            logger.warning(f"SerpApi search error: {response.status_code} — {response.text[:200]}")
            return []

        data = response.json()

        # Visual matches (pages with matching imagery)
        for item in data.get("visual_matches", []):
            url = item.get("link")
            if url:
                urls.append(url)

        # Knowledge graph links if present
        for item in data.get("knowledge_graph", []):
            url = item.get("link")
            if url:
                urls.append(url)

        logger.info(f"SerpApi (Google Lens): found {len(urls)} raw URLs")

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
    facecheck_scores: dict[str, float] = {}

    logger.info("=" * 50)
    logger.info("Starting multi-engine reverse image search")
    logger.info("=" * 50)

    # Engine 1: FaceCheck.ID (Specialized Face Biometrics for Social Media)
    if os.getenv("FACECHECK_API_KEY"):
        logger.info("\n[Engine 1/4] FaceCheck.ID (Social Biometrics)...")
        result["engines_queried"].append("facecheck")
        fc_urls, fc_scores = search_facecheck(primary_crop)
        if fc_urls:
            engine_results["facecheck"] = fc_urls
            facecheck_scores.update(fc_scores)
            result["engines_with_results"].append("facecheck")
            logger.info(f"FaceCheck.ID: {len(fc_urls)} biometric matches")
        else:
            logger.warning("FaceCheck.ID: 0 results")
        time.sleep(random.uniform(1.0, 2.0))
    else:
        logger.info("\n[Engine 1/4] FaceCheck.ID skipped (set FACECHECK_API_KEY in .env for Instagram/social search)")

    # Engine 2: Yandex (Primary Web Face Engine — Playwright Scraper)
    logger.info("\n[Engine 2/4] Yandex (Playwright)...")
    result["engines_queried"].append("yandex")
    yandex_urls = search_yandex(primary_crop)
    if yandex_urls:
        engine_results["yandex"] = yandex_urls
        result["engines_with_results"].append("yandex")
        logger.info(f"Yandex: {len(yandex_urls)} results")
    else:
        logger.warning("Yandex: 0 results")

    time.sleep(random.uniform(1.0, 2.0))

    # Engine 3: Bing Visual Search (Free Playwright Scraper or Azure API)
    logger.info("\n[Engine 3/4] Bing Visual Search...")
    result["engines_queried"].append("bing")
    bing_urls = search_bing(primary_crop)
    if bing_urls:
        engine_results["bing"] = bing_urls
        result["engines_with_results"].append("bing")
        logger.info(f"Bing: {len(bing_urls)} results")
    else:
        logger.warning("Bing: 0 results")

    time.sleep(random.uniform(1.0, 2.0))

    # Engine 4: Google Custom Search & SerpApi (fallback)
    all_results_so_far = sum(len(v) for v in engine_results.values())
    if all_results_so_far < 3:
        if os.getenv("GOOGLE_CSE_KEY"):
            logger.info("\n[Engine 4/4] Google Custom Search...")
            result["engines_queried"].append("google_cse")
            google_urls = search_google_custom(primary_crop)
            if google_urls:
                engine_results["google_cse"] = google_urls
                result["engines_with_results"].append("google_cse")
                logger.info(f"Google CSE: {len(google_urls)} results")

        if os.getenv("SERPAPI_KEY"):
            logger.info("\n[Engine 4/4] SerpApi (Google Reverse Image)...")
            result["engines_queried"].append("serpapi")
            serpapi_urls = search_serpapi(primary_crop)
            if serpapi_urls:
                engine_results["serpapi"] = serpapi_urls
                result["engines_with_results"].append("serpapi")
                logger.info(f"SerpApi: {len(serpapi_urls)} results")
            else:
                logger.warning("SerpApi: 0 results")
    else:
        logger.info("\n[Engine 4/4] SerpApi / Google CSE skipped (sufficient results from primary engines)")

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
        key=lambda u: (len(url_engine_map[u]), facecheck_scores.get(u, 0.0)),
        reverse=True
    )[:20]

    logger.info(f"\nScoring top {len(top_urls)} candidates...")

    for url in top_urls:
        engines_agreed = url_engine_map[url]
        engine_agreement = len(engines_agreed) / max(len(engine_results), 1)
        plat_score = platform_score(url)

        # pHash check
        phash_result = phash_similarity(face_crop_path, url)

        # Wayback Machine check
        wayback = {}
        if len(engines_agreed) >= 2:
            wayback = wayback_check(url)

        # FaceCheck biometric match confidence (if found via FaceCheck)
        facecheck_sim = facecheck_scores.get(url, 0.0)

        # Composite score calculation
        if facecheck_sim > 0:
            biometric_comp = facecheck_sim * 0.40
            agreement_comp = engine_agreement * 0.25
            platform_comp  = (plat_score / 10) * 0.20
            phash_comp     = 0.0
            if phash_result and phash_result.get("accessible"):
                dist = phash_result.get("hamming_distance", 64)
                phash_comp = max(0.0, 1.0 - dist / 64.0) * 0.15
            composite_score = biometric_comp + agreement_comp + platform_comp + phash_comp
        else:
            agreement_comp = engine_agreement * 0.45
            platform_comp  = (plat_score / 10) * 0.30
            phash_comp     = 0.0
            if phash_result and phash_result.get("accessible"):
                dist = phash_result.get("hamming_distance", 64)
                phash_comp = max(0.0, 1.0 - dist / 64.0) * 0.25
            composite_score = agreement_comp + platform_comp + phash_comp

        candidate = {
            "url": url,
            "engines": engines_agreed,
            "engine_agreement": round(engine_agreement, 3),
            "platform_score": plat_score,
            "platform": urlparse(url).netloc.replace("www.", ""),
            "phash": phash_result,
            "wayback": wayback,
            "facecheck_score": facecheck_sim,
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