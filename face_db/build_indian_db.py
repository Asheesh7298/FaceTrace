"""
face_db/build_indian_db.py
==========================
Auto-builds Indian tech/startup/public figure face database.
Downloads profile photos from Google CSE image search.

Categories:
  - Indian startup founders (100+ people)
  - Indian VCs and investors
  - Indian politicians (top 50)
  - Bollywood A-listers (top 50)
  - Indian tech executives
  - HH Goa ecosystem figures

Usage:
  python face_db/build_indian_db.py
"""

import os
import time
import random
import hashlib
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID")
OUTPUT_DIR     = Path("face_db/indian")

# ─── People Database ──────────────────────────────────────────────────────────

PEOPLE = {
    "startup_founders": [
        "Kunal Shah CRED founder",
        "Nithin Kamath Zerodha founder",
        "Harshil Mathur Razorpay CEO",
        "Shashvat Nakrani BharatPe founder",
        "Vijay Shekhar Sharma Paytm founder",
        "Byju Raveendran BYJU's founder",
        "Ritesh Agarwal OYO founder",
        "Gaurav Munjal Unacademy founder",
        "Sriharsha Majety Swiggy CEO",
        "Deepinder Goyal Zomato founder",
        "Binny Bansal Flipkart founder",
        "Sachin Bansal Flipkart founder",
        "Bhavish Aggarwal Ola founder",
        "Vidit Aatrey Meesho founder",
        "Ankush Sachdeva ShareChat founder",
        "Lalit Keshre Groww founder",
        "Ashneer Grover BharatPe founder",
        "Aadit Palicha Zepto founder",
        "Kaivalya Vohra Zepto founder",
        "Amrish Rau Pine Labs CEO",
        "Sameer Nigam PhonePe founder",
        "Naveen Tewari InMobi founder",
        "Saurabh Garg NoBroker founder",
        "Vamsi Krishna Vedantu founder",
        "Albinder Dhindsa Grofers founder",
        "Pankaj Chaddah Zomato cofounder",
        "Falguni Nayar Nykaa founder",
        "Radhakishan Damani DMart founder",
        "Divyank Turakhia Media.net founder",
        "Bhavin Turakhia Zeta founder",
    ],

    "investors_vcs": [
        "Sequoia India Shailendra Singh",
        "Accel India Subrata Mitra",
        "Accel India Prayank Sinha",
        "Blume Ventures Karthik Reddy",
        "Blume Ventures Sanjay Nath",
        "Matrix Partners Avnish Bajaj",
        "Nexus Venture Partners Naren Gupta",
        "Chiratae Ventures TC Meenakshisundaram",
        "Lightspeed India Bejul Somaia",
        "Elevation Capital Ravi Adusumalli",
        "Peak XV Partners GV Ravishankar",
        "3one4 Capital Pranav Pai",
        "India Quotient Anand Lunia",
        "Stellaris Venture Partners Ritesh Banglani",
        "Rainmatter Nithin Kamath",
        "Fundamentum Partnership Nandan Nilekani",
        "Jungle Ventures Vikram Bharati",
        "Y Combinator Garry Tan",
        "SoftBank India Sumer Juneja",
        "Tiger Global Scott Shleifer",
    ],

    "tech_executives": [
        "Sundar Pichai Google CEO",
        "Satya Nadella Microsoft CEO",
        "Shantanu Narayen Adobe CEO",
        "Arvind Krishna IBM CEO",
        "Sonia Syngal Gap CEO",
        "Leena Nair Chanel CEO",
        "Nandan Nilekani Infosys cofounder",
        "NR Narayana Murthy Infosys founder",
        "Azim Premji Wipro founder",
        "S Gopalakrishnan Infosys cofounder",
        "Kris Gopalakrishnan Infosys cofounder",
        "Salil Parekh Infosys CEO",
        "Thierry Delaporte Wipro CEO",
        "CP Gurnani Tech Mahindra CEO",
        "Rajesh Gopinathan TCS CEO",
        "Mukesh Aghi USISPF president",
        "Anant Maheshwari Microsoft India",
        "Sandeep Mathur Nokia India",
        "Vinod Dham Intel cofounder",
        "Sabeer Bhatia Hotmail founder",
    ],

    "politicians": [
        "Narendra Modi Prime Minister India",
        "Rahul Gandhi Congress president",
        "Arvind Kejriwal Delhi Chief Minister",
        "Mamata Banerjee West Bengal CM",
        "Yogi Adityanath UP Chief Minister",
        "Nirmala Sitharaman Finance Minister India",
        "S Jaishankar External Affairs Minister India",
        "Ashwini Vaishnaw IT Minister India",
        "Rajeev Chandrasekhar MoS IT India",
        "Smriti Irani BJP leader India",
        "Piyush Goyal Commerce Minister India",
        "Amit Shah Home Minister India",
        "Rajnath Singh Defence Minister India",
        "Shashi Tharoor Congress MP India",
        "Akhilesh Yadav SP leader India",
    ],

    "bollywood": [
        "Shah Rukh Khan actor Bollywood",
        "Amitabh Bachchan actor Bollywood",
        "Salman Khan actor Bollywood",
        "Aamir Khan actor Bollywood",
        "Priyanka Chopra actress Bollywood",
        "Deepika Padukone actress Bollywood",
        "Ranveer Singh actor Bollywood",
        "Ranbir Kapoor actor Bollywood",
        "Alia Bhatt actress Bollywood",
        "Katrina Kaif actress Bollywood",
        "Akshay Kumar actor Bollywood",
        "Hrithik Roshan actor Bollywood",
        "Vidya Balan actress Bollywood",
        "Kangana Ranaut actress Bollywood",
        "Anushka Sharma actress Bollywood",
    ],

    "startup_ecosystem": [
        "iSPIRT Sharad Sharma",
        "NASSCOM Debjani Ghosh",
        "Startup India Anurag Jain",
        "TiE Global Mahesh Murthy",
        "Antler India Nitin Sharma",
        "Entrepreneur First India",
        "GSF India Rajesh Sawhney",
        "LetsVenture Shanti Mohan",
        "AngelList India Utsav Somani",
        "Tracxn Neha Singh",
        "Venture Catalysts Apoorva Ranjan Sharma",
        "Mumbai Angels Nandini Mansinghka",
        "Indian Angel Network Saurabh Srivastava",
        "YourStory Shradha Sharma",
        "Inc42 Vaibhav Vardhan",
    ],
}


# ─── Downloader ───────────────────────────────────────────────────────────────

def download_person_photos(
    person_query: str,
    category: str,
    photos_per_person: int = 3,
) -> int:
    """
    Download profile photos for one person via Google CSE.
    Returns number of photos successfully downloaded.
    """
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logger.error("GOOGLE_API_KEY or GOOGLE_CSE_ID not set")
        return 0

    # Create folder named after person (first two words of query)
    person_name = "_".join(person_query.split()[:2]).lower()
    person_name = "".join(c for c in person_name if c.isalnum() or c == "_")
    person_dir = OUTPUT_DIR / category / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded enough photos
    existing = list(person_dir.glob("*.jpg"))
    if len(existing) >= photos_per_person:
        logger.info(f"  Already have {len(existing)} photos for {person_name}")
        return len(existing)

    downloaded = 0
    try:
        params = {
            "key":        GOOGLE_API_KEY,
            "cx":         GOOGLE_CSE_ID,
            "q":          f"{person_query} face portrait profile photo",
            "searchType": "image",
            "imgType":    "face",
            "num":        min(photos_per_person * 2, 10),
            "safe":       "active",
            "imgSize":    "medium",
        }

        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=10,
        )

        if resp.status_code != 200:
            logger.warning(f"  Google CSE error {resp.status_code} for {person_name}")
            return 0

        items = resp.json().get("items", [])
        if not items:
            logger.warning(f"  No results for {person_name}")
            return 0

        for i, item in enumerate(items):
            if downloaded >= photos_per_person:
                break

            img_url = item.get("link")
            if not img_url:
                continue

            try:
                headers = {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                    )
                }
                img_resp = requests.get(
                    img_url,
                    headers=headers,
                    timeout=8,
                    stream=True,
                )

                if img_resp.status_code != 200:
                    continue

                content_type = img_resp.headers.get("content-type", "")
                if "image" not in content_type:
                    continue

                img_data = img_resp.content
                if len(img_data) < 5000:  # skip tiny images
                    continue

                # Hash-based filename to avoid duplicates
                img_hash = hashlib.md5(img_data).hexdigest()[:8]
                img_path = person_dir / f"{i}_{img_hash}.jpg"

                with open(img_path, "wb") as f:
                    f.write(img_data)

                downloaded += 1
                logger.info(f"  ✓ {person_name} photo {downloaded}")

            except Exception as e:
                logger.debug(f"  Failed to download {img_url}: {e}")
                continue

            time.sleep(random.uniform(0.3, 0.8))

    except Exception as e:
        logger.warning(f"  Search failed for {person_name}: {e}")

    return downloaded


def build_database(photos_per_person: int = 3):
    """Build the complete Indian face database."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_people = sum(len(v) for v in PEOPLE.values())
    total_downloaded = 0
    processed = 0

    logger.info(f"Building Indian face database...")
    logger.info(f"Total people: {total_people}")
    logger.info(f"Photos per person: {photos_per_person}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 50)

    for category, people in PEOPLE.items():
        logger.info(f"\nCategory: {category} ({len(people)} people)")

        for person_query in people:
            processed += 1
            logger.info(f"[{processed}/{total_people}] {person_query}")

            n = download_person_photos(
                person_query,
                category,
                photos_per_person,
            )
            total_downloaded += n

            # Rate limit — Google CSE has 100 queries/day free
            time.sleep(random.uniform(1.0, 2.0))

    logger.info("\n" + "=" * 50)
    logger.info(f"Done. Downloaded {total_downloaded} photos for {total_people} people")
    logger.info(f"Database saved to: {OUTPUT_DIR}")

    # Print summary
    for category in PEOPLE.keys():
        cat_dir = OUTPUT_DIR / category
        if cat_dir.exists():
            people_dirs = [d for d in cat_dir.iterdir() if d.is_dir()]
            photos = list(cat_dir.rglob("*.jpg"))
            logger.info(f"  {category}: {len(people_dirs)} people, {len(photos)} photos")


def verify_database():
    """Verify database structure is correct for DeepFace."""
    logger.info("Verifying database structure...")

    issues = []
    for category_dir in OUTPUT_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        for person_dir in category_dir.iterdir():
            if not person_dir.is_dir():
                continue
            photos = list(person_dir.glob("*.jpg"))
            if len(photos) == 0:
                issues.append(f"Empty: {person_dir}")
            elif len(photos) == 1:
                logger.warning(f"Only 1 photo: {person_dir.name}")

    if issues:
        logger.warning(f"Issues found: {len(issues)}")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("Database structure is valid ✓")

    total_photos = list(OUTPUT_DIR.rglob("*.jpg"))
    total_people = [
        d for d in OUTPUT_DIR.rglob("*")
        if d.is_dir() and not any(p.is_dir() for p in d.iterdir()
        if p.is_dir())
    ]

    logger.info(f"Total photos: {len(total_photos)}")


if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        verify_database()
    else:
        photos = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        build_database(photos_per_person=photos)
        verify_database()