"""
search/yandex.py
===============
Yandex reverse-image search via Playwright (free, no API key). Yandex is the best
free engine at finding *other photos of the same person* across the web. Fragile
(layout changes / CAPTCHA), so it is a best-effort booster: we filter its results
to social/profile domains and hand them to the pooled ranker.

Requires: pip install playwright && python -m playwright install chromium
Windows: the WindowsSelectorEventLoopPolicy must be set (pipeline.py / app.py do this).
"""

import time
import random
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Only keep leads on domains that indicate a real profile/post.
_KEEP = ("instagram.com", "x.com", "twitter.com", "facebook.com", "linkedin.com",
         "youtube.com", "tiktok.com", "imdb.com", "wikipedia.org")


def _is_useful(url: str) -> bool:
    d = urlparse(url).netloc.lower()
    return any(k in d for k in _KEEP)


def search_yandex(image_path: str, timeout_ms: int = 20000) -> list[dict]:
    """Reverse-image search; returns [{url, source:'yandex', title}] on social domains."""
    leads = []
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            ctx = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"),
                viewport={"width": 1280, "height": 800},
            )
            page = ctx.new_page()
            page.goto("https://yandex.com/images/", timeout=timeout_ms)
            time.sleep(random.uniform(1.0, 2.0))

            # Open "search by image"
            for sel in ('button[aria-label*="image" i]', '.cbir-icon',
                        '[class*="cbir"]', 'button.search-by-image'):
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0:
                        loc.first.click(timeout=3000)
                        break
                except Exception:
                    continue
            time.sleep(random.uniform(0.5, 1.0))

            fi = page.locator('input[type="file"]')
            if fi.count() == 0:
                logger.warning("Yandex: file input not found (CAPTCHA/layout change)")
                browser.close()
                return []
            fi.first.set_input_files(str(Path(image_path).resolve()), timeout=5000)
            time.sleep(random.uniform(3.0, 5.0))

            anchors = page.eval_on_selector_all(
                "a[href]", "els => els.map(el => ({href: el.href, text: el.innerText}))"
            )
            seen = set()
            for a in anchors:
                href = a.get("href", "")
                if href.startswith("http") and _is_useful(href) and href not in seen:
                    seen.add(href)
                    leads.append({"url": href, "source": "yandex",
                                  "title": (a.get("text") or "").strip()[:120]})
            browser.close()
            logger.info(f"Yandex: {len(leads)} social leads")
    except Exception as e:
        logger.warning(f"Yandex failed: {e}")
    return leads[:15]


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m search.yandex <image_path>")
        sys.exit(1)
    print(json.dumps(search_yandex(sys.argv[1]), indent=2))
