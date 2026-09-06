"""
search/bing.py
=============
Bing Visual Search via Playwright (free, no key) — EXPERIMENTAL. Uploads the
image and scrapes the "pages that include this image" results, filtered to
useful social/news domains. Fragile (layout changes / bot checks), so it's
opt-in via ENABLE_BING=1 and never blocks the pipeline on failure.

Requires: playwright + chromium. Windows: WindowsSelectorEventLoopPolicy
(set by pipeline.py / app.py).

Returns {"names": [], "leads": [{url, title}, ...]}.
"""

import time
import random
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_KEEP = ("instagram.com", "x.com", "twitter.com", "facebook.com", "linkedin.com",
         "youtube.com", "github.com", "imdb.com", "wikipedia.org", "devfolio.co",
         "yourstory.com", "inc42.com", "forbes.com")


def _useful(url: str) -> bool:
    d = urlparse(url).netloc.lower()
    return any(k in d for k in _KEEP)


def search_bing(image_path: str, timeout_ms: int = 20000) -> dict:
    out = {"names": [], "leads": []}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, args=["--no-sandbox", "--disable-blink-features=AutomationControlled"])
            ctx = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"),
                viewport={"width": 1280, "height": 800})
            page = ctx.new_page()
            page.goto("https://www.bing.com/images", timeout=timeout_ms)
            time.sleep(random.uniform(1.0, 2.0))
            # open the visual-search / camera control
            for sel in ('a#sb_sbip', '.camera', '[aria-label*="Visual Search" i]',
                        '[title*="Visual Search" i]'):
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
                browser.close()
                logger.warning("Bing: file input not found")
                return out
            fi.first.set_input_files(str(Path(image_path).resolve()), timeout=5000)
            time.sleep(random.uniform(4.0, 6.0))
            anchors = page.eval_on_selector_all(
                "a[href]", "els => els.map(el => ({href: el.href, text: el.innerText}))")
            seen = set()
            for a in anchors:
                href = a.get("href", "")
                if href.startswith("http") and _useful(href) and href not in seen:
                    seen.add(href)
                    out["leads"].append({"url": href, "title": (a.get("text") or "").strip()[:120]})
            browser.close()
            logger.info(f"Bing: {len(out['leads'])} useful leads")
    except Exception as e:
        logger.warning(f"Bing failed: {e}")
    out["leads"] = out["leads"][:15]
    return out


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print(json.dumps(search_bing(sys.argv[1]), indent=2))
