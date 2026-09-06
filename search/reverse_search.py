"""
search/reverse_search.py
========================
Stage 2 — Identity-first reverse search.

Pipeline (why this order):
  1. Host the face crop at a public URL (Google Lens needs a URL, not a file).
  2. Google Lens  -> ai_overview -> extract the person's NAME.
     (Lens `visual_matches` are look-alikes, NOT identity — we do not trust them.)
  3. Name-based social search -> real LinkedIn / X / Instagram / etc. profiles.
  4. VERIFY-BEFORE-CLAIM: download an official photo of the identified person
     and DeepFace.verify() it against the input crop. We only claim a match
     when the face actually verifies — this eliminates false positives.
  5. Download the matched image bytes and hash THEM (not the URL string) so the
     blockchain record is a real tamper-evident fingerprint of the content.

Every external call degrades gracefully: a missing key or a failed engine
reduces confidence but never crashes the pipeline.
"""

import os
import io
import hashlib
import logging
import requests
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from search.image_host import host_image
from search.identity import extract_person_name, _person_via_spacy, _person_via_regex
from search.facecheck import search_facecheck, facecheck_available
from search.curated_db import search_curated_db, lookup_name
from search.yandex import search_yandex
from search.vision import search_vision, vision_available
from search.rekognition import search_rekognition, rekognition_available
from search.bing import search_bing
from search.tineye import search_tineye, tineye_available

logger = logging.getLogger(__name__)

# A FaceCheck biometric score at/above this is a confident match on its own
# (FaceCheck already did the face matching), even without a name to verify.
FACECHECK_STRONG = 0.70

SERPAPI = "https://serpapi.com/search"

# Social platforms ranked by how much a hit there proves a real profile.
PLATFORM_PRIORITY = {
    "instagram.com": 10, "x.com": 10, "twitter.com": 10, "linkedin.com": 10,
    "facebook.com": 9, "youtube.com": 8, "github.com": 8, "devfolio.co": 8,
    "imdb.com": 7, "wikipedia.org": 7, "yourstory.com": 7, "crunchbase.com": 7,
    "dev.to": 7, "behance.net": 7,
    "inc42.com": 6, "wellfound.com": 6, "researchgate.net": 6, "medium.com": 5,
    "reddit.com": 5, "pinterest.com": 3,
    # News/press — recognized + scored LOW so it never outranks a real profile,
    # but adds corroboration + a datable, archivable web result.
    "forbes.com": 4, "techcrunch.com": 4, "entrackr.com": 4, "bloomberg.com": 4,
    "economictimes.indiatimes.com": 3, "business-standard.com": 3,
    "thehindu.com": 3, "hindustantimes.com": 3, "ndtv.com": 3,
    "timesofindia.indiatimes.com": 3, "moneycontrol.com": 3,
}

# Platforms to target in the name-based search (all bundled into ONE combined
# query, so adding sites costs no extra API calls). Social/dev first (the real
# deliverable), then a few press outlets for corroboration.
SOCIAL_SITES = [
    "linkedin.com", "instagram.com", "x.com", "twitter.com", "facebook.com",
    "github.com", "devfolio.co", "reddit.com", "dev.to", "imdb.com",
]
NEWS_SITES = [
    "yourstory.com", "inc42.com", "forbes.com", "techcrunch.com", "entrackr.com",
]

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FaceTrace/1.0"}


def _noop(*_a, **_k):
    pass


def platform_score(url: str) -> int:
    domain = urlparse(url).netloc.replace("www.", "")
    for platform, score in PLATFORM_PRIORITY.items():
        if platform in domain:
            return score
    return 2


def platform_name(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


# ─── SerpApi: Google Lens (identity) ─────────────────────────────────────────

def _name_from_kg(kg: Optional[dict]) -> Optional[str]:
    """Name from a Google Lens knowledge_graph entry, when it's clearly a person."""
    if not kg:
        return None
    title = kg.get("title")
    t = (kg.get("type") or "").lower()
    person_hints = ("actor", "actress", "singer", "musician", "player", "rapper",
                    "politician", "director", "model", "author", "comedian",
                    "athlete", "footballer", "cricketer", "businessperson", "ceo",
                    "entrepreneur", "youtuber", "personality", "dancer", "producer")
    if title and (any(h in t for h in person_hints)
                  or _person_via_spacy(title) or _person_via_regex(title)):
        return title
    return None


def _name_from_visual(vms: list) -> Optional[str]:
    """Most-repeated PERSON name across visual-match titles (needs corroboration)."""
    from collections import Counter
    c = Counter()
    for m in vms[:20]:
        nm = _name_from_title(m.get("title", ""))
        if nm:
            c[nm] += 1
    for nm, cnt in c.most_common(1):
        if cnt >= 2:      # appears on 2+ pages → more trustworthy
            return nm
    return None


def _one_lens_call(hosted_url: str, api_key: str):
    d = requests.get(SERPAPI, params={
        "engine": "google_lens", "url": hosted_url, "api_key": api_key,
    }, timeout=45).json()
    vms = d.get("visual_matches", []) or []
    kg = d.get("knowledge_graph")
    ao = d.get("ai_overview")
    if ao and ao.get("page_token"):     # expand the AI overview
        ai = requests.get(SERPAPI, params={
            "engine": "google_ai_overview",
            "page_token": ao["page_token"], "api_key": api_key,
        }, timeout=45).json()
        ao = ai.get("ai_overview", ao)
    return vms, kg, ao


def lens_identify(hosted_url: str, api_key: str, retries: int = 1) -> dict:
    """Robust Google Lens identity. The ai_overview is stochastic, so we retry it,
    then fall back to the knowledge_graph and to a corroborated name mined from
    visual-match titles. Returns {name, ai_overview, visual_matches}."""
    out = {"name": None, "ai_overview": None, "visual_matches": []}
    if not api_key or not hosted_url:
        return out
    best_vms, best_kg, best_ao = [], None, None
    for attempt in range(retries + 1):
        try:
            vms, kg, ao = _one_lens_call(hosted_url, api_key)
        except Exception as e:
            logger.warning(f"Lens call failed (attempt {attempt + 1}): {e}")
            continue
        if len(vms) > len(best_vms):
            best_vms = vms
        best_kg = kg or best_kg
        best_ao = ao or best_ao
        name = extract_person_name(ao)
        if name:
            out.update({"name": name, "ai_overview": ao, "visual_matches": vms})
            logger.info(f"Lens: identity '{name}' (ai_overview, attempt {attempt + 1})")
            return out
    # ai_overview gave no name across retries → try fallbacks
    out["visual_matches"], out["ai_overview"] = best_vms, best_ao
    out["name"] = _name_from_kg(best_kg) or _name_from_visual(best_vms)
    logger.info(f"Lens: {len(best_vms)} visual matches | identity: "
                f"{(out['name'] + ' (fallback)') if out['name'] else 'not identified'}")
    return out


# ─── SerpApi: name-based social profile search ───────────────────────────────

def name_social_search(name: str, api_key: str) -> list[dict]:
    """One combined Google query across social sites. Returns [{url, title}]."""
    if not name or not api_key:
        return []
    sites = " OR ".join(f"site:{s}" for s in (SOCIAL_SITES + NEWS_SITES))
    query = f'"{name}" ({sites})'
    results = []
    try:
        d = requests.get(SERPAPI, params={
            "engine": "google", "q": query, "num": 20, "api_key": api_key,
        }, timeout=30).json()
        for item in d.get("organic_results", []) or []:
            link = item.get("link")
            if link:
                results.append({"url": link, "title": item.get("title", "")})
        logger.info(f"Name search '{name}': {len(results)} profile candidates")
    except Exception as e:
        logger.warning(f"Name search failed: {e}")
    return results


# ─── Verify-before-claim ─────────────────────────────────────────────────────

# How many independent reference photos must match before we CLAIM an identity.
# Requiring agreement across several official photos is what stops Google Lens
# look-alike misidentifications from being claimed.
MIN_REF_MATCHES = 2
MAX_REFS_TO_CHECK = 6


def _reference_photo_urls(name: str, api_key: str, k: int = 10) -> list[str]:
    """Fetch several candidate reference-photo URLs of `name` via Google Images."""
    if not api_key:
        return []
    urls = []
    try:
        d = requests.get(SERPAPI, params={
            "engine": "google_images", "q": f"{name} face closeup portrait",
            "num": 20, "api_key": api_key,
        }, timeout=30).json()
        for item in d.get("images_results", []) or []:
            u = item.get("original")
            if u and u.startswith("http"):
                urls.append(u)
            if len(urls) >= k:
                break
    except Exception as e:
        logger.warning(f"Reference-photo lookup failed: {e}")
    return urls


def verify_identity(name: str, face_crop_path: str, api_key: str,
                    output_dir: str = "output") -> dict:
    """
    Verify `name` against the input crop using MULTIPLE reference photos.

    An identity is only CONFIRMED when at least MIN_REF_MATCHES independent
    official photos of `name` match the input face. This rejects Google Lens
    look-alike misidentifications (references of the wrong person won't match).

    Returns {verified, score, matches, refs_checked, reference_url, reference_path}.
    score = 1 - best_cosine_distance (higher = more similar).
    """
    result = {"verified": False, "score": 0.0, "matches": 0, "refs_checked": 0,
              "reference_url": None, "reference_path": None}
    ref_urls = _reference_photo_urls(name, api_key)
    if not ref_urls:
        logger.warning("No reference photos found — cannot verify identity")
        return result

    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from deepface import DeepFace

    # ArcFace + cosine: DeepFace's own verification threshold is 0.68 (distance).
    THRESH = 0.68

    def _embed(path):
        try:
            r = DeepFace.represent(img_path=path, model_name="ArcFace",
                                   detector_backend="retinaface", enforce_detection=False)
            v = np.array(r[0]["embedding"], dtype=np.float64)
            n = np.linalg.norm(v)
            return v / n if n > 0 else None
        except Exception:
            return None

    # Embed the query crop ONCE (verify() used to re-embed it for every reference).
    q_emb = _embed(face_crop_path)
    if q_emb is None:
        logger.warning("Could not embed input crop for verification")
        return result

    # Download references in PARALLEL (network I/O is the other slow part).
    def _fetch(url):
        try:
            resp = requests.get(url, headers=_UA, timeout=12)
            if resp.status_code == 200 and "image" in resp.headers.get("content-type", "") \
                    and len(resp.content) >= 3000:
                return url, resp.content
        except Exception:
            pass
        return url, None

    refs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for url, content in ex.map(_fetch, ref_urls[:MAX_REFS_TO_CHECK + 3]):
            if content:
                refs.append((url, content))

    best_distance, best_url, best_bytes = 1.0, None, None
    matches = checked = 0
    for url, content in refs:
        if checked >= MAX_REFS_TO_CHECK:
            break
        try:
            tmp = str(Path(output_dir) / "_ref_tmp.jpg")
            Image.open(io.BytesIO(content)).convert("RGB").save(tmp, quality=95)
        except Exception:
            continue
        r_emb = _embed(tmp)
        if r_emb is None:
            continue
        checked += 1
        dist = 1.0 - float(np.dot(q_emb, r_emb))  # cosine distance
        if dist < THRESH:
            matches += 1
        if dist < best_distance:
            best_distance, best_url, best_bytes = dist, url, content
        logger.info(f"  ref {checked}: dist={dist:.3f} "
                    f"{'match' if dist < THRESH else 'no'} ({url[:45]})")
        # Early-stop: enough agreement to CONFIRM — no need to check the rest.
        if matches >= MIN_REF_MATCHES:
            break

    result["refs_checked"] = checked
    result["matches"] = matches
    result["score"] = round(1 - best_distance, 3) if checked else 0.0
    result["verified"] = matches >= MIN_REF_MATCHES

    # Persist the best-matching reference for the UI side-by-side comparison.
    if best_url and best_bytes:
        result["reference_url"] = best_url
        try:
            ref_path = str(Path(output_dir) / "reference_photo.jpg")
            Image.open(io.BytesIO(best_bytes)).convert("RGB").save(ref_path, quality=95)
            result["reference_path"] = ref_path
        except Exception:
            pass

    logger.info(f"Identity verification: {name} -> "
                f"{'CONFIRMED' if result['verified'] else 'NOT confirmed'} "
                f"({matches}/{checked} references matched, best score {result['score']})")
    return result


# ─── Content hashing (real bytes, not the URL string) ────────────────────────

def download_and_hash(url: str, output_dir: str = "output") -> Optional[dict]:
    """Download an image and hash its bytes. Returns None if not retrievable."""
    try:
        resp = requests.get(url, headers=_UA, timeout=15)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "image" not in ctype or len(resp.content) < 1000:
            return None
        digest = "sha256:" + hashlib.sha256(resp.content).hexdigest()
        saved = str(Path(output_dir) / "matched_content.jpg")
        try:
            Image.open(io.BytesIO(resp.content)).convert("RGB").save(saved, quality=95)
        except Exception:
            saved = None
        return {"content_hash": digest, "bytes": len(resp.content),
                "content_type": ctype, "image_url": url, "saved_path": saved}
    except Exception as e:
        logger.warning(f"Content download failed for {url}: {e}")
        return None


# ─── Forensic signal: Wayback first-seen ─────────────────────────────────────

def wayback_check(url: str) -> dict:
    result = {"has_archive": False, "first_seen": None}
    try:
        cdx = (f"https://web.archive.org/cdx/search/cdx?url={url}"
               f"&output=json&limit=1&fl=timestamp&fastLatest=true")
        resp = requests.get(cdx, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1:
                ts = data[1][0]
                result = {"has_archive": True,
                          "first_seen": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"}
    except Exception:
        pass
    return result


# ─── Helpers for pooling ─────────────────────────────────────────────────────

def _name_from_title(title: str) -> Optional[str]:
    if not title:
        return None
    return _person_via_spacy(title) or _person_via_regex(title)


def _hash_b64_thumb(b64: str, output_dir: str) -> Optional[dict]:
    """Hash a FaceCheck base64 thumbnail as the matched-content fingerprint."""
    try:
        import base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        if len(raw) < 400:
            return None
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        saved = str(Path(output_dir) / "matched_content.jpg")
        with open(saved, "wb") as f:
            f.write(raw)
        return {"content_hash": digest, "bytes": len(raw),
                "content_type": "image/jpeg (FaceCheck thumbnail)",
                "image_url": "facecheck_thumbnail", "saved_path": saved}
    except Exception:
        return None


# ─── Candidate scoring (pooled, biometric-aware) ─────────────────────────────

# A person's own PROFILE page (what we want) vs a POST/article that merely mentions
# them. Profiles have shallow paths (/handle, /in/x, /@x); posts are deep.
_PROFILE_HINTS = ("/in/", "/@")
_POST_HINTS = ("/p/", "/posts/", "/reel/", "/status/", "/activity-", "/watch",
               "/pulse/", "/photo", "/video", "/news/", "/story", "/article")


def _is_profile(url: str) -> tuple[bool, bool]:
    """Return (is_profile, is_post)."""
    path = urlparse(url).path.rstrip("/").lower()
    is_post = any(h in path for h in _POST_HINTS)
    if is_post:
        return False, True
    if any(h in path for h in _PROFILE_HINTS):
        return True, False
    segs = [s for s in path.split("/") if s]
    return len(segs) <= 1, False      # a single shallow segment = a profile page


def score_leads(leads: list[dict], confirmed_name: Optional[str],
                verification_score: float) -> list[dict]:
    name_parts = [p.lower() for p in (confirmed_name or "").split() if len(p) > 2]
    scored, seen = [], set()
    for c in leads:
        url = c.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        plat = platform_score(url)
        title = (c.get("title") or "").lower()
        bio = float(c.get("biometric") or 0.0)
        name_in = any(p in url.lower() or p in title for p in name_parts)
        is_profile, is_post = _is_profile(url)
        # A profile page is the real deliverable → boost it; a post that only
        # mentions the person → slight penalty so it never beats the real profile.
        profile_adj = 0.14 if is_profile else (-0.06 if is_post else 0.0)
        composite = (bio * 0.34 + (plat / 10) * 0.28 + (0.18 if name_in else 0.0)
                     + verification_score * 0.12 + profile_adj)
        scored.append({
            "url": url,
            "title": c.get("title", ""),
            "platform": platform_name(url),
            "platform_score": plat,
            "biometric": round(bio, 3),
            "name_in_result": name_in,
            "is_profile": is_profile,
            "engines": [c.get("engine", "?")],
            "composite_score": round(max(0.0, composite), 4),
        })
    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


# ─── Orchestrator (multi-engine: FaceCheck + Lens + Curated DB + Yandex) ─────

def multi_engine_search(face_crop_path: str, output_dir: str = "output",
                        exif_data: Optional[dict] = None,
                        progress: Callable = None,
                        identify_image_path: Optional[str] = None) -> dict:
    """
    Runs all available engines, pools their leads, resolves an identity with
    verify-before-claim, and picks the best matching social URL.

    identify_image_path: image used for identification (Lens/FaceCheck/Yandex).
    Defaults to the crop, but the ORIGINAL photo identifies far better. The face
    crop is always what gets face-verified and hashed.
    """
    from collections import Counter
    progress = progress or _noop
    api_key = os.getenv("SERPAPI_KEY")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lens_image = identify_image_path or face_crop_path

    result = {
        "success": False,
        "hosted_crop_url": None,
        "person_name": None,          # CONFIRMED identity (verified) — what we claim
        "lens_suggested_name": None,  # top unconfirmed candidate (for honest UI)
        "name_verified": False,
        "verification_score": 0.0,
        "verification_matches": 0,
        "verification_refs": 0,
        "verification_photo_url": None,
        "engines_queried": [],
        "engines_with_results": [],
        "best_match": None,
        "all_candidates": [],
        "total_candidates": 0,
        "matched_content": None,
        "lens_visual_matches_count": 0,
    }

    leads: list[dict] = []
    name_votes: Counter = Counter()

    def q(e):
        if e not in result["engines_queried"]:
            result["engines_queried"].append(e)

    def hit(e):
        if e not in result["engines_with_results"]:
            result["engines_with_results"].append(e)

    def add_name(nm, w):
        if nm and len(nm.split()) >= 2:
            name_votes[nm.strip()] += w

    # ── ENGINES 1-4 run in PARALLEL (independent I/O + one GPU task) ──────────
    # Only the curated-DB task touches DeepFace/TF here (Lens/FaceCheck/Yandex are
    # pure network/browser), so there is no concurrent-TF hazard.
    from concurrent.futures import ThreadPoolExecutor

    def _t_facecheck():
        return search_facecheck(lens_image) if facecheck_available() else None

    def _t_lens():
        h = host_image(lens_image)
        l = {"name": None, "visual_matches": []}
        if h and api_key:
            l = lens_identify(h, api_key)      # retries + kg/visual fallbacks inside
        # If the original photo yielded no name, try the tight face crop too.
        if api_key and not l.get("name") and face_crop_path != lens_image:
            h2 = host_image(face_crop_path)
            if h2:
                l2 = lens_identify(h2, api_key)
                if l2.get("name") or len(l2.get("visual_matches", [])) > len(l.get("visual_matches", [])):
                    l, h = l2, (h or h2)
        return h, l

    def _t_db():
        return search_curated_db(face_crop_path)

    def _t_yandex():
        if os.getenv("ENABLE_YANDEX") != "1":
            return None
        try:
            return search_yandex(lens_image)
        except Exception as e:
            logger.warning(f"Yandex booster failed: {e}")
            return []

    # Extra pluggable engines — each returns {"names":[(nm,w)], "leads":[...]}, or
    # None when not configured. They activate only when their key/flag is set, so
    # they add zero latency and zero risk when unconfigured.
    def _t_vision():
        return search_vision(face_crop_path) if vision_available() else None

    def _t_rekognition():
        return search_rekognition(face_crop_path) if rekognition_available() else None

    def _t_bing():
        if os.getenv("ENABLE_BING") != "1":
            return None
        try:
            return search_bing(lens_image)
        except Exception as e:
            logger.warning(f"Bing booster failed: {e}")
            return None

    def _t_tineye():
        return search_tineye(face_crop_path) if tineye_available() else None

    progress("search_step", {"step": "engines",
                             "message": "Running all identity engines in parallel"})
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_fc, f_lens, f_db, f_yx = (ex.submit(_t_facecheck), ex.submit(_t_lens),
                                    ex.submit(_t_db), ex.submit(_t_yandex))
        f_vis, f_rek, f_bing, f_tin = (ex.submit(_t_vision), ex.submit(_t_rekognition),
                                       ex.submit(_t_bing), ex.submit(_t_tineye))
        fc = f_fc.result()
        hosted, lens = f_lens.result()
        db = f_db.result()
        yx = f_yx.result()
        extra = {"google_vision": f_vis.result(), "aws_rekognition": f_rek.result(),
                 "bing": f_bing.result(), "tineye": f_tin.result()}

    # Merge FaceCheck
    fc_best = None
    if fc is not None:
        q("facecheck")
        if fc:
            hit("facecheck")
            fc_best = fc[0]
        for l in fc:
            leads.append({"url": l["url"], "engine": "facecheck",
                          "biometric": l["score"], "title": "", "thumb": l.get("thumb")})

    # Merge Lens
    result["hosted_crop_url"] = hosted
    result["lens_visual_matches_count"] = len(lens["visual_matches"])
    if hosted and api_key:
        q("google_lens")
    if lens["name"]:
        hit("google_lens")
        add_name(lens["name"], 2.5)
        progress("identity", {"name": lens["name"], "status": "candidate"})
    for m in lens["visual_matches"]:
        link = m.get("link")
        if link and platform_score(link) >= 7:
            leads.append({"url": link, "engine": "lens_visual", "title": m.get("title", "")})
            add_name(_name_from_title(m.get("title", "")), 0.4)

    # Merge Curated DB (may carry a social URL straight from Wikidata)
    q("curated_db")
    if db.get("found"):
        hit("curated_db")
        add_name(db["name"], 1.0 + db["score"])
        for u in (db.get("instagram"), db.get("twitter")):
            if u:
                leads.append({"url": u, "engine": "curated_db", "title": db["name"]})

    # Merge Yandex
    if yx is not None:
        q("yandex")
        if yx:
            hit("yandex")
        for l in yx:
            leads.append({"url": l["url"], "engine": "yandex", "title": l.get("title", "")})
            add_name(_name_from_title(l.get("title", "")), 0.4)

    # Merge extra pluggable engines uniformly ({"names":[(nm,w)], "leads":[...]}).
    for eng_name, res in extra.items():
        if res is None:
            continue
        q(eng_name)
        got = bool(res.get("names") or res.get("leads"))
        if got:
            hit(eng_name)
        for nm, w in res.get("names", []):
            add_name(nm, w)
        for l in res.get("leads", []):
            leads.append({"url": l["url"], "engine": eng_name, "title": l.get("title", "")})
            add_name(_name_from_title(l.get("title", "")), 0.3)

    # ── RESOLVE IDENTITY — verify top candidates (verify-before-claim) ────────
    ranked = [n for n, _ in name_votes.most_common()]
    result["lens_suggested_name"] = ranked[0] if ranked else None

    # Speculatively fetch social profiles for the top candidate WHILE we verify
    # (name_search is pure network, verify is TF — no conflict). Saves ~3s when
    # the top candidate is the one that verifies (the common case).
    _prof_ex = ThreadPoolExecutor(max_workers=1)
    prof_future, spec_name = None, None
    if ranked and api_key:
        spec_name = ranked[0]
        prof_future = _prof_ex.submit(name_social_search, spec_name, api_key)

    confirmed = None
    best_ver = {"verified": False, "score": 0.0, "matches": 0, "refs_checked": 0,
                "reference_url": None}
    for cand in ranked[:2]:
        if not api_key:
            break
        progress("search_step", {"step": "verify",
                                 "message": f"Verifying '{cand}' against reference photos"})
        ver = verify_identity(cand, face_crop_path, api_key, str(output_dir))
        if ver["score"] > best_ver["score"]:
            best_ver = ver
        if ver["verified"]:
            confirmed = cand
            best_ver = ver
            break

    result["name_verified"] = best_ver["verified"]
    result["verification_score"] = best_ver["score"]
    result["verification_matches"] = best_ver["matches"]
    result["verification_refs"] = best_ver["refs_checked"]
    result["verification_photo_url"] = best_ver["reference_url"]
    result["person_name"] = confirmed

    if confirmed:
        progress("verification", {"verified": True, "score": best_ver["score"],
                                  "matches": best_ver["matches"], "refs": best_ver["refs_checked"],
                                  "suggested": confirmed, "reference_url": best_ver["reference_url"]})
    elif ranked:
        progress("verification", {"verified": False, "score": best_ver["score"],
                                  "matches": best_ver["matches"], "refs": best_ver["refs_checked"],
                                  "suggested": result["lens_suggested_name"],
                                  "reference_url": best_ver["reference_url"]})

    # ── Gazetteer: if the confirmed name is in our 18k index, grab its stored
    #    Instagram/Twitter directly (pure name lookup, no face match → 0 risk).
    if confirmed:
        gaz = lookup_name(confirmed)
        if gaz:
            hit("curated_db")
            for u in (gaz.get("instagram"), gaz.get("twitter")):
                if u:
                    leads.append({"url": u, "engine": "curated_db", "title": confirmed})

    # ── If confirmed, fetch clean profile URLs by name (reuse the speculative
    #    search when it was for the confirmed name; else do a fresh one) ────────
    if confirmed and api_key:
        progress("search_step", {"step": "profiles", "message": f"Finding social profiles for {confirmed}"})
        q("name_search")
        profiles = None
        if prof_future is not None and spec_name == confirmed:
            try:
                profiles = prof_future.result(timeout=30)
            except Exception:
                profiles = None
        if profiles is None:
            profiles = name_social_search(confirmed, api_key)
        if profiles:
            hit("name_search")
        for p in profiles:
            leads.append({"url": p["url"], "engine": "name_search", "title": p.get("title", "")})
    # release the speculative executor
    _prof_ex.shutdown(wait=False)

    # ── SCORE + pick best ─────────────────────────────────────────────────────
    scored = score_leads(leads, confirmed, result["verification_score"])
    result["all_candidates"] = scored
    result["total_candidates"] = len(scored)

    fc_strong = bool(fc_best and fc_best.get("score", 0) >= FACECHECK_STRONG)

    if scored:
        best = scored[0]
        best["wayback"] = wayback_check(best["url"])
        result["best_match"] = best
        # Claimable = verified identity with a real profile, OR a strong FaceCheck match.
        result["success"] = bool(
            (confirmed and result["name_verified"] and best["platform_score"] >= 6) or fc_strong
        )

        # Content fingerprint: verified reference photo → else FaceCheck thumbnail.
        content = None
        if result["name_verified"] and result["verification_photo_url"]:
            content = download_and_hash(result["verification_photo_url"], str(output_dir))
        if not content and fc_best and fc_best.get("thumb"):
            content = _hash_b64_thumb(fc_best["thumb"], str(output_dir))
        if content:
            result["matched_content"] = content

        if result["success"]:
            progress("match", {"url": best["url"], "platform": best["platform"],
                               "score": best["composite_score"]})
            logger.info(f"MATCH: {best['url']} ({best['platform']}) "
                        f"[identity: {confirmed or 'FaceCheck biometric'}]")
        else:
            logger.info("Leads found but not confidently verified — proof-of-scan record")

    return result


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m search.reverse_search <face_crop_path>")
        sys.exit(1)
    r = multi_engine_search(sys.argv[1])
    printable = {k: v for k, v in r.items() if k != "all_candidates"}
    printable["candidate_count"] = r["total_candidates"]
    print(json.dumps(printable, indent=2, default=str))
