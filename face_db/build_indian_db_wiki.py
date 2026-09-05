"""
face_db/build_indian_db_wiki.py
================================
Builds Indian face DB using Wikipedia images directly.
No API key needed — uses Wikipedia's free image API.

Usage:
    python face_db/build_indian_db_wiki.py
"""

import os
import time
import random
import requests
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("face_db/indian")

# Wikipedia page titles for each person
PEOPLE = {
    "startup_founders": [
        ("kunal_shah",        "Kunal Shah (entrepreneur)"),
        ("nithin_kamath",     "Nithin Kamath"),
        ("deepinder_goyal",   "Deepinder Goyal"),
        ("bhavish_aggarwal",  "Bhavish Aggarwal"),
        ("vijay_shekhar",     "Vijay Shekhar Sharma"),
        ("ritesh_agarwal",    "Ritesh Agarwal"),
        ("sachin_bansal",     "Sachin Bansal"),
        ("binny_bansal",      "Binny Bansal"),
        ("falguni_nayar",     "Falguni Nayar"),
        ("ashneer_grover",    "Ashneer Grover"),
        ("byju_raveendran",   "Byju Raveendran"),
        ("naveen_tewari",     "Naveen Tewari"),
        ("sameer_nigam",      "Sameer Nigam"),
        ("gaurav_munjal",     "Gaurav Munjal"),
    ],
    "tech_executives": [
        ("sundar_pichai",     "Sundar Pichai"),
        ("satya_nadella",     "Satya Nadella"),
        ("shantanu_narayen",  "Shantanu Narayen"),
        ("arvind_krishna",    "Arvind Krishna (businessman)"),
        ("nandan_nilekani",   "Nandan Nilekani"),
        ("narayana_murthy",   "N. R. Narayana Murthy"),
        ("azim_premji",       "Azim Premji"),
        ("sabeer_bhatia",     "Sabeer Bhatia"),
        ("vinod_dham",        "Vinod Dham"),
        ("kris_gopalakrishnan","Kris Gopalakrishnan"),
    ],
    "politicians": [
        ("narendra_modi",     "Narendra Modi"),
        ("rahul_gandhi",      "Rahul Gandhi"),
        ("arvind_kejriwal",   "Arvind Kejriwal"),
        ("mamata_banerjee",   "Mamata Banerjee"),
        ("nirmala_sitharaman","Nirmala Sitharaman"),
        ("s_jaishankar",      "S. Jaishankar"),
        ("amit_shah",         "Amit Shah"),
        ("shashi_tharoor",    "Shashi Tharoor"),
        ("smriti_irani",      "Smriti Irani"),
        ("yogi_adityanath",   "Yogi Adityanath"),
    ],
    "bollywood": [
        ("shah_rukh_khan",    "Shah Rukh Khan"),
        ("amitabh_bachchan",  "Amitabh Bachchan"),
        ("salman_khan",       "Salman Khan"),
        ("aamir_khan",        "Aamir Khan"),
        ("priyanka_chopra",   "Priyanka Chopra"),
        ("deepika_padukone",  "Deepika Padukone"),
        ("ranveer_singh",     "Ranveer Singh"),
        ("alia_bhatt",        "Alia Bhatt"),
        ("hrithik_roshan",    "Hrithik Roshan"),
        ("akshay_kumar",      "Akshay Kumar"),
    ],
}


def get_wikipedia_image(page_title: str) -> str | None:
    """Get main image URL from a Wikipedia page."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + \
              requests.utils.quote(page_title)
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "FaceTrace-DB-Builder/1.0 (educational project)"
        })
        if resp.status_code != 200:
            return None
        data = resp.json()
        thumbnail = data.get("thumbnail", {})
        original  = data.get("originalimage", {})
        # Prefer original, fall back to thumbnail
        img_url = original.get("source") or thumbnail.get("source")
        return img_url
    except Exception as e:
        logger.debug(f"Wikipedia API error for {page_title}: {e}")
        return None


def download_image(url: str, save_path: Path) -> bool:
    """Download image from URL."""
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "FaceTrace-DB-Builder/1.0 (educational project)"
        })
        if resp.status_code != 200:
            return False
        if len(resp.content) < 5000:
            return False
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type:
            return False
        save_path.write_bytes(resp.content)
        return True
    except Exception as e:
        logger.debug(f"Download error: {e}")
        return False


def build_database():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(len(v) for v in PEOPLE.values())
    downloaded = 0
    processed  = 0

    logger.info(f"Building Indian face DB from Wikipedia...")
    logger.info(f"Total people: {total}")
    logger.info("=" * 50)

    for category, people in PEOPLE.items():
        logger.info(f"\nCategory: {category}")

        for slug, wiki_title in people:
            processed += 1
            person_dir = OUTPUT_DIR / category / slug
            person_dir.mkdir(parents=True, exist_ok=True)

            # Skip if already downloaded
            existing = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
            if existing:
                logger.info(f"  [{processed}/{total}] {slug} — already have {len(existing)} photos")
                downloaded += len(existing)
                continue

            logger.info(f"  [{processed}/{total}] {wiki_title}...")

            img_url = get_wikipedia_image(wiki_title)
            if not img_url:
                logger.warning(f"  No Wikipedia image found for {wiki_title}")
                time.sleep(0.5)
                continue

            # Determine extension
            ext = ".jpg"
            if ".png" in img_url.lower():
                ext = ".png"
            elif ".jpeg" in img_url.lower():
                ext = ".jpg"

            save_path = person_dir / f"0_wiki{ext}"
            success   = download_image(img_url, save_path)

            if success:
                downloaded += 1
                logger.info(f"  ✓ Downloaded {wiki_title}")
            else:
                logger.warning(f"  ✗ Failed to download {wiki_title}")

            time.sleep(random.uniform(0.3, 0.8))

    logger.info("\n" + "=" * 50)
    logger.info(f"Done. {downloaded} photos downloaded for {total} people")

    # Summary
    for category in PEOPLE.keys():
        cat_dir = OUTPUT_DIR / category
        if cat_dir.exists():
            photos = list(cat_dir.rglob("*.jpg")) + list(cat_dir.rglob("*.png"))
            logger.info(f"  {category}: {len(photos)} photos")


if __name__ == "__main__":
    build_database()
