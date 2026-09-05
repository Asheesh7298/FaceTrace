"""
search/curated_db.py
====================
Our own local face-search index — an ArcFace embedding index over a curated set
of findable public figures, queried locally in <100ms (no cloud dependency at
run time). Built properly (detect + align + ArcFace), so it never produces the
false positives the old skip-detector DB did.

A DB match only PROPOSES a name candidate; verify-before-claim confirms it. But
because entries carry the person's Instagram/Twitter (straight from Wikidata),
a confirmed match yields the real social URL with no extra search.

Seeding is FREE via Wikidata (name + image P18 + Instagram P2003 + Twitter P2002):
    python -m search.curated_db wikidata 800      # build ~800 from Wikidata
    python -m search.curated_db build             # build from the name seed list
    python -m search.curated_db add "Name"
    python -m search.curated_db search crop.jpg
    python -m search.curated_db stats

For large-scale builds, embed on Modal then download the index — see
face_index_modal.py. The runtime query always stays local.
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

INDEX_PATH = Path("face_db/curated_index.json")          # small local builds (JSON)
NPZ_PATH = Path("face_db/curated_index.npz")             # large builds (binary float32)
META_PATH = Path("face_db/curated_meta.json")            # metadata for the npz
_UA = {"User-Agent": "FaceTrace/1.0 (educational hackathon project; contact via github)"}

# Similarity to even PROPOSE a candidate (verify-before-claim confirms afterwards).
# ArcFace cosine similarity of ~0.5 is NOT reliable at scale — in an 18k gallery
# some look-alike always scores ~0.5 (Tom Hanks -> Rich Eisen). So for a large
# index we demand a genuinely CONFIDENT match (0.60+) and a margin over the
# runner-up. This makes the DB high-precision: it stays silent unless it's sure,
# and Google Lens handles everyone else. Verify-before-claim is the final gate.
MIN_SIM = 0.35
MIN_SIM_LARGE = 0.60
LARGE_INDEX = 5000
MARGIN = 0.05

WD_SPARQL = "https://query.wikidata.org/sparql"

# Country QIDs for optional filtering (India first — HH Goa ecosystem).
COUNTRY_INDIA = "Q668"

DEFAULT_SEED = [
    "Sundar Pichai", "Satya Nadella", "Elon Musk", "Bill Gates", "Sam Altman",
    "Narayana Murthy", "Nandan Nilekani", "Ratan Tata", "Mukesh Ambani",
    "Kunal Shah", "Bhavish Aggarwal", "Ritesh Agarwal", "Vijay Shekhar Sharma",
    "Narendra Modi", "Shashi Tharoor", "Virat Kohli", "MS Dhoni",
    "Alia Bhatt", "Deepika Padukone", "Priyanka Chopra", "Tom Hanks",
    "Emma Watson", "Cristiano Ronaldo", "Barack Obama",
]


# ─── Wikidata sourcing (name + image + socials, free) ────────────────────────

def fetch_wikidata_people(limit: int = 500, offset: int = 0,
                          country_qid: Optional[str] = None,
                          require_instagram: bool = True,
                          min_sitelinks: Optional[int] = None) -> list[dict]:
    """People (humans) with a photo. Returns [{name, wikidata_id, image_url,
    instagram, twitter}].

    require_instagram=True  -> only people who have an Instagram handle (direct social).
    require_instagram=False -> any notable person with a photo; `min_sitelinks`
                               gates by fame (# of Wikipedia languages). Social, if
                               any, still captured; else the pipeline name-searches it.
    """
    country = f"; wdt:P27 wd:{country_qid}" if country_qid else ""
    lines = [f"?p wdt:P31 wd:Q5 ; wdt:P18 ?img {country} ."]
    # Requiring Instagram keeps the set bounded (~101k) so the query stays fast.
    # An image-only query with a sitelinks filter times out on the public endpoint.
    if require_instagram:
        lines.append("?p wdt:P2003 ?ig .")
    else:
        lines.append("OPTIONAL { ?p wdt:P2003 ?ig . }")
    if min_sitelinks:
        lines.append("?p wikibase:sitelinks ?sl .")
        lines.append(f"FILTER(?sl >= {min_sitelinks})")
    lines.append("OPTIONAL { ?p wdt:P2002 ?tw . }")
    lines.append('SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }')
    q = (f"SELECT ?p ?pLabel ?img ?ig ?tw WHERE {{ {' '.join(lines)} }} "
         f"LIMIT {limit} OFFSET {offset}")
    try:
        r = requests.get(WD_SPARQL, params={"query": q, "format": "json"},
                         headers=_UA, timeout=90)
        if r.status_code != 200:
            logger.warning(f"Wikidata query failed: {r.status_code}")
            return []
        out, seen = [], set()
        for b in r.json()["results"]["bindings"]:
            qid = b["p"]["value"].rsplit("/", 1)[-1]
            if qid in seen:
                continue
            seen.add(qid)
            name = b["pLabel"]["value"]
            if name.startswith("Q") and name[1:].isdigit():
                continue  # unlabeled entity
            ig = b.get("ig", {}).get("value", "").lstrip("@")
            tw = b.get("tw", {}).get("value", "").lstrip("@")
            img = b["img"]["value"]
            if "?" not in img:
                img += "?width=512"   # thumbnail, not full-res
            out.append({
                "name": name, "wikidata_id": qid, "image_url": img,
                "instagram": f"https://www.instagram.com/{ig}/" if ig else None,
                "twitter": f"https://x.com/{tw}" if tw else None,
            })
        return out
    except Exception as e:
        logger.warning(f"Wikidata fetch error: {e}")
        return []


# ─── Embedding ───────────────────────────────────────────────────────────────

def _embed(image_path: str) -> Optional[np.ndarray]:
    try:
        from deepface import DeepFace
        reps = DeepFace.represent(img_path=image_path, model_name="ArcFace",
                                  detector_backend="retinaface", enforce_detection=False)
        if not reps:
            return None
        v = np.array(reps[0]["embedding"], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None
    except Exception as e:
        logger.debug(f"embed failed for {image_path}: {e}")
        return None


def _embed_url(url: str, tmp="face_db/_tmp_ref.jpg") -> Optional[np.ndarray]:
    try:
        import io
        from PIL import Image
        r = requests.get(url, headers=_UA, timeout=15)
        if r.status_code != 200 or len(r.content) < 2000:
            return None
        Path(tmp).parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(r.content)).convert("RGB").save(tmp, quality=95)
        return _embed(tmp)
    except Exception:
        return None


# ─── Index build / load ──────────────────────────────────────────────────────

def load_index(path: Path = INDEX_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(entries: list[dict], path: Path = INDEX_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (temp + replace) so a concurrent reader never sees a partial file.
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(entries), encoding="utf-8")
    os.replace(tmp, path)


# Cached matrix so we don't reparse the index on every scan.
_CACHE = {"emb": None, "meta": None, "sig": None}


def save_npz(embeddings: np.ndarray, meta: list[dict]):
    """Persist a LARGE index as binary float32 (+ JSON metadata). Used by the
    Modal builder. Embeddings must already be L2-normalized."""
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PATH, emb=embeddings.astype(np.float32))
    META_PATH.write_text(json.dumps(meta), encoding="utf-8")


def load_matrix():
    """Return (embeddings float32 [N,512], meta list). Prefers the npz index;
    falls back to the JSON index. Cached by file signature."""
    if NPZ_PATH.exists() and META_PATH.exists():
        sig = (NPZ_PATH.stat().st_mtime, META_PATH.stat().st_mtime)
        if _CACHE["sig"] != sig:
            _CACHE["emb"] = np.load(NPZ_PATH)["emb"].astype(np.float32)
            _CACHE["meta"] = json.loads(META_PATH.read_text(encoding="utf-8"))
            _CACHE["sig"] = sig
        return _CACHE["emb"], _CACHE["meta"]
    entries = load_index()
    if not entries:
        return None, []
    sig = ("json", INDEX_PATH.stat().st_mtime if INDEX_PATH.exists() else 0, len(entries))
    if _CACHE["sig"] != sig:
        _CACHE["emb"] = np.array([e["embedding"] for e in entries], dtype=np.float32)
        _CACHE["meta"] = [{k: v for k, v in e.items() if k != "embedding"} for e in entries]
        _CACHE["sig"] = sig
    return _CACHE["emb"], _CACHE["meta"]


def _wiki_image_url(name: str) -> Optional[str]:
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ', '_')}",
                         headers=_UA, timeout=12)
        if r.status_code != 200:
            return None
        d = r.json()
        return (d.get("originalimage") or {}).get("source") or (d.get("thumbnail") or {}).get("source")
    except Exception:
        return None


def add_person(name: str, path: Path = INDEX_PATH, url: Optional[str] = None,
               instagram: Optional[str] = None, twitter: Optional[str] = None,
               wikidata_id: Optional[str] = None) -> bool:
    entries = load_index(path)
    if any(e["name"].lower() == name.lower() for e in entries):
        return True
    img_url = url or _wiki_image_url(name)
    if not img_url:
        logger.warning(f"No image for '{name}'")
        return False
    emb = _embed_url(img_url)
    if emb is None:
        logger.warning(f"Could not embed '{name}'")
        return False
    entries.append({"name": name, "image_url": img_url, "embedding": emb.tolist(),
                    "instagram": instagram, "twitter": twitter, "wikidata_id": wikidata_id})
    _save_index(entries, path)
    return True


def build_index(names: Optional[list[str]] = None, path: Path = INDEX_PATH):
    names = names or DEFAULT_SEED
    ok = 0
    for i, name in enumerate(names, 1):
        logger.info(f"[{i}/{len(names)}] {name}")
        if add_person(name, path):
            ok += 1
    logger.info(f"Curated index: {ok}/{len(names)} added")


def build_from_wikidata(target: int = 800, path: Path = INDEX_PATH,
                        country_qid: Optional[str] = None, page: int = 300):
    """Page through Wikidata, embed each person's photo, store with socials."""
    entries = load_index(path)
    have = {e.get("wikidata_id") for e in entries if e.get("wikidata_id")}
    added = 0
    offset = 0
    while added < target:
        people = fetch_wikidata_people(limit=page, offset=offset, country_qid=country_qid)
        if not people:
            break
        offset += page
        for p in people:
            if added >= target:
                break
            if p["wikidata_id"] in have:
                continue
            emb = _embed_url(p["image_url"])
            if emb is None:
                continue
            entries.append({
                "name": p["name"], "image_url": p["image_url"],
                "embedding": emb.tolist(), "instagram": p["instagram"],
                "twitter": p["twitter"], "wikidata_id": p["wikidata_id"],
            })
            have.add(p["wikidata_id"])
            added += 1
            if added % 25 == 0:
                _save_index(entries, path)
                logger.info(f"  …{added} added (total {len(entries)})")
    _save_index(entries, path)
    logger.info(f"Wikidata build done: +{added} (index now {len(entries)})")


# ─── Search ──────────────────────────────────────────────────────────────────

def search_curated_db(crop_path: str, path: Path = INDEX_PATH,
                      min_sim: Optional[float] = None) -> dict:
    """Best candidate from the local index (a NAME + its socials to confirm).

    For large indexes, requires a higher similarity AND a margin over the
    runner-up, so a random look-alike twin doesn't get proposed.
    """
    result = {"found": False, "name": None, "score": 0.0,
              "image_url": None, "instagram": None, "twitter": None}
    emb, meta = load_matrix()
    if emb is None or not len(meta):
        return result
    q = _embed(crop_path)
    if q is None:
        return result
    q = q.astype(np.float32)

    n = len(meta)
    thresh = min_sim if min_sim is not None else (MIN_SIM_LARGE if n >= LARGE_INDEX else MIN_SIM)
    sims = emb @ q
    best = int(np.argmax(sims))
    best_sim = float(sims[best])
    # margin over the runner-up (only meaningful on large indexes)
    margin_ok = True
    if n >= LARGE_INDEX:
        second = float(np.partition(sims, -2)[-2]) if n > 1 else 0.0
        margin_ok = (best_sim - second) >= MARGIN

    if best_sim >= thresh and margin_ok:
        e = meta[best]
        result.update({"found": True, "name": e["name"], "score": round(best_sim, 3),
                       "image_url": e.get("image_url"), "instagram": e.get("instagram"),
                       "twitter": e.get("twitter")})
        logger.info(f"Curated DB candidate: {result['name']} (sim {result['score']}, "
                    f"index {n})")
    else:
        logger.info(f"Curated DB: no confident candidate (best {best_sim:.3f}, index {n})")
    return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "wikidata":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else 800
        # India first (most relevant), then global
        build_from_wikidata(target=target, country_qid=COUNTRY_INDIA)
        build_from_wikidata(target=target, country_qid=None)
    elif cmd == "build":
        build_index()
    elif cmd == "add" and len(sys.argv) >= 3:
        add_person(" ".join(sys.argv[2:]))
    elif cmd == "search" and len(sys.argv) >= 3:
        print(json.dumps(search_curated_db(sys.argv[2]), indent=2))
    elif cmd == "stats":
        idx = load_index()
        with_ig = sum(1 for e in idx if e.get("instagram"))
        print(f"Index: {len(idx)} people | {with_ig} with Instagram")
    else:
        print("Usage: python -m search.curated_db [wikidata <n> | build | add <name> | search <crop> | stats]")
