"""
face_index_modal.py — OPTIONAL large-scale index builder on Modal
=================================================================
Builds a LARGE curated face index by embedding thousands of Wikidata people's
photos in parallel on Modal GPUs, then writes the compact index locally. The
pipeline then queries that index LOCALLY (<100ms) — Modal is only used to BUILD,
never at run time, so there is no cloud dependency during the live demo.

This is optional. A few-hundred-to-few-thousand person index built locally with
`python -m search.curated_db wikidata 800` is already strong and needs no Modal.

Setup (once):
    pip install modal
    modal token new          # logs into YOUR Modal account (free tier is plenty)

TEST SMALL FIRST (verify it works end-to-end on your account before a huge run):
    modal run face_index_modal.py --ig-target 200 --total-target 300

Then scale up (Tier 1 = everyone with Instagram, Tier 2 = fame-ranked fill):
    modal run face_index_modal.py --ig-target 50000 --total-target 100000

Output: face_db/curated_index.npz + face_db/curated_meta.json
        (written locally; the pipeline queries them LOCALLY — no Modal at run time)
"""

import json
import modal

app = modal.App("facetrace-index")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    # tensorflow[and-cuda] bundles the NVIDIA CUDA + cuDNN wheels so TF actually
    # uses the GPU (Modal supplies the driver). Plain `tensorflow` falls back to CPU.
    .pip_install("deepface==0.0.93", "tf-keras", "tensorflow[and-cuda]==2.21.0",
                 "opencv-python-headless==4.9.0.80", "numpy==1.26.4", "Pillow==10.3.0",
                 "requests", "retina-face")
)

_UA = {"User-Agent": "FaceTrace/1.0 (educational hackathon project)"}


def _download(url, ua, tries=4):
    """Download an image with retry/backoff (Wikimedia throttles bursts)."""
    import time, requests
    for i in range(tries):
        try:
            r = requests.get(url, headers=ua, timeout=25)
            if r.status_code == 200 and len(r.content) >= 2000:
                return r.content
            if r.status_code in (429, 503, 500):     # throttled — back off
                time.sleep(1.5 * (i + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 * (i + 1))
    return None


@app.function(image=image, gpu="T4", timeout=1800, retries=1)
def embed_batch(people: list):
    """Embed a BATCH of people in one container: load the model once, then
    download+embed each sequentially. Batching keeps concurrency low so Wikimedia
    doesn't rate-limit us (the #1 cause of 'undetectable' failures), and reuses the
    ArcFace model across the whole batch. Runs on a T4 GPU."""
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import io, numpy as np
    from PIL import Image
    from deepface import DeepFace
    # Descriptive UA (Wikimedia policy) reduces the chance of being blocked.
    ua = {"User-Agent": "FaceTraceBot/1.0 (educational hackathon project; "
                        "https://github.com/; contact via github)"}
    out = []
    for p in people:
        content = _download(p["image_url"], ua)
        if not content:
            continue
        try:
            tmp = "/tmp/f.jpg"
            Image.open(io.BytesIO(content)).convert("RGB").save(tmp, quality=95)
            reps = DeepFace.represent(img_path=tmp, model_name="ArcFace",
                                      detector_backend="retinaface", enforce_detection=False)
            if not reps:
                continue
            v = np.array(reps[0]["embedding"], dtype=np.float64)
            n = np.linalg.norm(v)
            if n == 0:
                continue
            out.append({"name": p["name"], "image_url": p["image_url"],
                        "instagram": p.get("instagram"), "twitter": p.get("twitter"),
                        "wikidata_id": p.get("wikidata_id"), "embedding": (v / n).tolist()})
        except Exception:
            continue
    return out


def _fetch_wikidata(limit, offset, country_qid=None, require_instagram=True, min_sitelinks=None):
    import requests
    country = f"; wdt:P27 wd:{country_qid}" if country_qid else ""
    lines = [f"?p wdt:P31 wd:Q5 ; wdt:P18 ?img {country} ."]
    if require_instagram:
        lines.append("?p wdt:P2003 ?ig .")
    else:
        lines.append("?p wikibase:sitelinks ?sl .")
        lines.append("OPTIONAL { ?p wdt:P2003 ?ig . }")
        if min_sitelinks:
            lines.append(f"FILTER(?sl >= {min_sitelinks})")
    lines.append("OPTIONAL { ?p wdt:P2002 ?tw . }")
    lines.append('SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }')
    q = f"SELECT ?p ?pLabel ?img ?ig ?tw WHERE {{ {' '.join(lines)} }} LIMIT {limit} OFFSET {offset}"
    r = requests.get("https://query.wikidata.org/sparql",
                     params={"query": q, "format": "json"}, headers=_UA, timeout=120)
    if r.status_code != 200:
        return []
    out = []
    for b in r.json()["results"]["bindings"]:
        qid = b["p"]["value"].rsplit("/", 1)[-1]
        name = b["pLabel"]["value"]
        if name.startswith("Q") and name[1:].isdigit():
            continue
        ig = b.get("ig", {}).get("value", "").lstrip("@")
        tw = b.get("tw", {}).get("value", "").lstrip("@")
        img = b["img"]["value"]
        img += ("?width=512" if "?" not in img else "")
        out.append({"name": name, "wikidata_id": qid, "image_url": img,
                    "instagram": f"https://www.instagram.com/{ig}/" if ig else None,
                    "twitter": f"https://x.com/{tw}" if tw else None})
    return out


def _collect(target, seen, require_instagram, min_sitelinks, country=None, page=400):
    people, offset, empties = [], 0, 0
    while len(people) < target:
        batch = _fetch_wikidata(page, offset, country, require_instagram, min_sitelinks)
        if not batch:
            empties += 1
            if empties >= 3:      # tolerate transient 504s, then give up this stream
                break
            continue
        empties = 0
        offset += page
        for p in batch:
            if p["wikidata_id"] not in seen:
                seen.add(p["wikidata_id"])
                people.append(p)
                if len(people) >= target:
                    break
    return people


@app.local_entrypoint()
def main(target: int = 18000, global_sitelinks: int = 0):
    """
    Fame-relevant build (recommended ~15-20k), sourced from Wikidata's Instagram
    set (fast + everyone has a direct social link):
      1. ALL Indian people with Instagram + photo (most relevant for HH Goa).
      2. Global people with Instagram + photo to fill to `target`. Set
         --global-sitelinks N (>0) to prefer more famous people (slower fetch).
    Embeds on Modal GPUs; saves a compact .npz index locally, queried LOCALLY.

      modal run face_index_modal.py --target 18000
      modal run face_index_modal.py --target 18000 --global-sitelinks 6   # famous-only
    """
    import numpy as np, os
    seen = set()
    print("Fetching ALL Indian public figures with Instagram…")
    people = _collect(target, seen, require_instagram=True, min_sitelinks=None, country="Q668")
    print(f"  Indian: {len(people)}")
    if len(people) < target:
        floor = global_sitelinks or None
        print(f"Filling with global (Instagram{', sitelinks>=' + str(global_sitelinks) if floor else ''}) to {target}…")
        people += _collect(target - len(people), seen, require_instagram=True,
                           min_sitelinks=floor, country=None)
    print(f"Total {len(people)} candidates — embedding on Modal in batches…")

    # Batch so ~N containers download sequentially (gentle on Wikimedia) instead
    # of thousands hitting it at once. Each batch reuses the loaded ArcFace model.
    BATCH = 150
    batches = [people[i:i + BATCH] for i in range(0, len(people), BATCH)]
    entries = []
    for res in embed_batch.map(batches):
        entries.extend(res)
    with_ig = sum(1 for e in entries if e.get("instagram"))
    print(f"Embedded {len(entries)} faces ({with_ig} with a direct Instagram); "
          f"skipped {len(people) - len(entries)} (download/detect failures).")

    emb = np.array([e.pop("embedding") for e in entries], dtype=np.float32)
    os.makedirs("face_db", exist_ok=True)
    np.savez_compressed("face_db/curated_index.npz", emb=emb)
    with open("face_db/curated_meta.json", "w", encoding="utf-8") as f:
        json.dump(entries, f)
    print(f"Saved face_db/curated_index.npz + curated_meta.json ({len(entries)} people). "
          f"Query stays LOCAL — no Modal at run time.")
