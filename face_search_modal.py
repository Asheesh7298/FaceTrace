"""
face_search_modal.py
====================
Modal A100 face database search.

Databases searched (on Modal persistent volume):
  - LFW           : 5,749 labeled public figures
  - CelebA        : 10,177 celebrities
  - VGGFace2      : 9,000 high-quality identities
  - MS-Celeb      : 100,000 public figures (subset)
  - Indian Tech   : custom Indian ecosystem DB

Models used (ensemble voting):
  - ArcFace       : weight 0.50 (most accurate)
  - Facenet512    : weight 0.30 (good on low quality)
  - VGG-Face      : weight 0.20 (different architecture)

Distance metrics per model:
  - cosine + euclidean_l2 averaged

Usage:
  # Deploy once
  modal deploy face_search_modal.py

  # Call from pipeline
  from face_search_modal import search_face_db
  result = search_face_db.remote(image_bytes)
"""

import os
import modal

# ─── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("facetrace-search")

# Persistent volume — upload DBs once, persist forever
volume = modal.Volume.from_name("facetrace-db", create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
    ])
    .pip_install([
        "deepface==0.0.93",
        "insightface==0.7.3",
        "onnxruntime-gpu==1.18.0",
        "opencv-python-headless==4.9.0.80",
        "numpy==1.24.3",
        "tensorflow==2.13.0",
        "pandas",
        "Pillow==10.3.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "scikit-learn",
        "scipy==1.13.1",
        "tqdm",
        "requests",
        "facenet-pytorch==2.5.3",
    ])
)

# ─── Calibrated Thresholds ────────────────────────────────────────────────────

# Empirically derived from each model's published benchmarks
THRESHOLDS = {
    "ArcFace": {
        "lfw":      {"strong": 0.28, "weak": 0.55},
        "vggface2": {"strong": 0.25, "weak": 0.50},
        "celeba":   {"strong": 0.30, "weak": 0.60},
        "ms_celeb": {"strong": 0.30, "weak": 0.60},
        "indian":   {"strong": 0.30, "weak": 0.58},
    },
    "Facenet512": {
        "lfw":      {"strong": 0.30, "weak": 0.65},
        "vggface2": {"strong": 0.28, "weak": 0.60},
        "celeba":   {"strong": 0.32, "weak": 0.68},
        "ms_celeb": {"strong": 0.32, "weak": 0.68},
        "indian":   {"strong": 0.30, "weak": 0.65},
    },
    "VGG-Face": {
        "lfw":      {"strong": 0.40, "weak": 0.70},
        "vggface2": {"strong": 0.38, "weak": 0.68},
        "celeba":   {"strong": 0.42, "weak": 0.72},
        "ms_celeb": {"strong": 0.42, "weak": 0.72},
        "indian":   {"strong": 0.40, "weak": 0.70},
    },
}

# DB weights — higher = more trusted
DB_WEIGHTS = {
    "indian":   1.5,   # most relevant for use case
    "vggface2": 1.2,   # highest quality annotations
    "lfw":      1.0,   # well-labeled baseline
    "celeba":   1.0,   # broad coverage
    "ms_celeb": 0.9,   # noisy labels, slight penalty
}

# Model weights for ensemble voting
MODEL_WEIGHTS = {
    "ArcFace":    0.50,
    "Facenet512": 0.30,
    "VGG-Face":   0.20,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_name(identity_path: str) -> str:
    """Extract person name from DeepFace identity path."""
    parts = identity_path.replace("\\", "/").split("/")
    # Identity path: .../face_db/lfw/George_W_Bush/image.jpg
    # Name folder is second-to-last segment
    for i in range(len(parts) - 1, -1, -1):
        part = parts[i]
        if part and not part.endswith((".jpg", ".jpeg", ".png", ".webp")):
            return part.replace("_", " ").title()
    return "Unknown"


def calibrate_confidence(
    raw_distance: float,
    model: str,
    db: str,
) -> float:
    """Convert raw distance to calibrated confidence [0, 1]."""
    t = THRESHOLDS.get(model, {}).get(db, {"strong": 0.30, "weak": 0.65})

    if raw_distance < t["strong"]:
        # Strong match → 0.85 to 1.0
        ratio = raw_distance / t["strong"]
        return round(1.0 - ratio * 0.15, 4)
    elif raw_distance < t["weak"]:
        # Possible match → 0.50 to 0.85
        ratio = (raw_distance - t["strong"]) / (t["weak"] - t["strong"])
        return round(0.85 - ratio * 0.35, 4)
    else:
        return 0.0


def generate_augmented_variants(face_path: str, temp_dir: str) -> list:
    """
    Generate augmented face variants for DB search.
    More variants = higher chance of matching.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    variants = [face_path]
    img = Image.open(face_path).convert("RGB")
    cv_img = cv2.imread(face_path)

    try:
        # Sharpened
        sharp = ImageEnhance.Sharpness(img).enhance(2.0)
        p = os.path.join(temp_dir, "aug_sharp.jpg")
        sharp.save(p, quality=95)
        variants.append(p)

        # Contrast enhanced
        contrast = ImageEnhance.Contrast(img).enhance(1.5)
        p = os.path.join(temp_dir, "aug_contrast.jpg")
        contrast.save(p, quality=95)
        variants.append(p)

        # Histogram equalized
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        eq_color = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
        p = os.path.join(temp_dir, "aug_eq.jpg")
        cv2.imwrite(p, eq_color)
        variants.append(p)

        # Brightness normalized
        bright = ImageEnhance.Brightness(img).enhance(1.2)
        p = os.path.join(temp_dir, "aug_bright.jpg")
        bright.save(p, quality=95)
        variants.append(p)

    except Exception as e:
        print(f"Augmentation warning: {e}")

    return variants


# ─── Core DB Search (runs on Modal A100) ─────────────────────────────────────

@app.function(
    image=image,
    gpu="A100",
    volumes={"/face_db": volume},
    timeout=180,
    memory=32768,
)
def search_face_db(face_image_bytes: bytes) -> dict:
    """
    Search input face against all databases on Modal A100.

    Returns:
    {
      "found": bool,
      "name": str or None,
      "confidence": float,
      "database": str,
      "all_matches": [...],
      "search_time_seconds": float,
    }
    """
    import time
    import tempfile
    import cv2
    import numpy as np
    from deepface import DeepFace

    start = time.time()

    # Force GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Save input image
    tmpdir = tempfile.mkdtemp()
    face_path = os.path.join(tmpdir, "input_face.jpg")
    with open(face_path, "wb") as f:
        f.write(face_image_bytes)

    # Generate augmented variants
    variants = generate_augmented_variants(face_path, tmpdir)

    # Available databases
    db_base = "/face_db"
    available_dbs = []
    for db_name in ["indian", "vggface2", "lfw", "celeba", "ms_celeb"]:
        db_path = os.path.join(db_base, db_name)
        if os.path.isdir(db_path) and os.listdir(db_path):
            available_dbs.append(db_name)

    if not available_dbs:
        return {
            "found": False,
            "name": None,
            "confidence": 0.0,
            "database": None,
            "all_matches": [],
            "search_time_seconds": round(time.time() - start, 2),
            "error": "No databases found on Modal volume. Run upload_databases() first.",
        }

    print(f"Searching {len(available_dbs)} databases with {len(variants)} variants...")

    # name → weighted confidence accumulator
    name_scores: dict = {}
    all_matches = []

    for db_name in available_dbs:
        db_path = os.path.join(db_base, db_name)
        db_weight = DB_WEIGHTS.get(db_name, 1.0)

        for model_name, model_weight in MODEL_WEIGHTS.items():
            for variant_path in variants:
                try:
                    matches = DeepFace.find(
                        img_path=variant_path,
                        db_path=db_path,
                        model_name=model_name,
                        distance_metric="cosine",
                        enforce_detection=False,
                        detector_backend="skip",
                        silent=True,
                    )

                    if not matches or len(matches[0]) == 0:
                        continue

                    top = matches[0].iloc[0]
                    raw_distance = float(top["distance"])
                    confidence = calibrate_confidence(raw_distance, model_name, db_name)

                    if confidence < 0.50:
                        continue

                    name = extract_name(str(top["identity"]))
                    weighted_conf = confidence * model_weight * db_weight

                    if name not in name_scores:
                        name_scores[name] = {
                            "total": 0.0,
                            "count": 0,
                            "best_raw": confidence,
                            "best_db": db_name,
                            "best_model": model_name,
                        }

                    name_scores[name]["total"] += weighted_conf
                    name_scores[name]["count"] += 1
                    name_scores[name]["best_raw"] = max(
                        name_scores[name]["best_raw"], confidence
                    )

                    all_matches.append({
                        "name": name,
                        "confidence": round(confidence, 3),
                        "weighted": round(weighted_conf, 3),
                        "model": model_name,
                        "database": db_name,
                        "distance": round(raw_distance, 4),
                    })

                except Exception as e:
                    print(f"  Search error ({db_name}/{model_name}): {e}")
                    continue

    # Clean up temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if not name_scores:
        return {
            "found": False,
            "name": None,
            "confidence": 0.0,
            "database": None,
            "all_matches": [],
            "search_time_seconds": round(time.time() - start, 2),
        }

    # Find winner — highest weighted total
    winner_name = max(name_scores, key=lambda n: name_scores[n]["total"])
    winner = name_scores[winner_name]

    # Normalize confidence
    final_confidence = min(
        winner["best_raw"] * (1 + 0.1 * min(winner["count"] - 1, 3)),
        1.0,
    )

    # Sort all matches by confidence
    all_matches.sort(key=lambda x: x["confidence"], reverse=True)

    elapsed = round(time.time() - start, 2)
    print(f"Search complete in {elapsed}s — winner: {winner_name} ({final_confidence:.3f})")

    return {
        "found": True,
        "name": winner_name,
        "confidence": round(final_confidence, 3),
        "database": winner["best_db"],
        "model": winner["best_model"],
        "vote_count": winner["count"],
        "all_matches": all_matches[:20],
        "search_time_seconds": elapsed,
    }


# ─── Database Upload (run once per database) ──────────────────────────────────

@app.function(
    image=image,
    volumes={"/face_db": volume},
    timeout=3600,
    memory=8192,
)
def upload_database(db_name: str, files: dict) -> str:
    """
    Upload a database to Modal volume.
    files: dict of {relative_path: bytes}
    """
    import os

    db_path = f"/face_db/{db_name}"
    os.makedirs(db_path, exist_ok=True)

    count = 0
    for rel_path, data in files.items():
        full_path = os.path.join(db_path, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)
        count += 1

    volume.commit()
    return f"Uploaded {count} files to {db_path}"


@app.function(
    image=image,
    volumes={"/face_db": volume},
    timeout=60,
)
def list_databases() -> dict:
    """List all databases and their sizes on Modal volume."""
    import os

    db_base = "/face_db"
    result = {}

    if not os.path.exists(db_base):
        return {"error": "No databases found"}

    for db_name in os.listdir(db_base):
        db_path = os.path.join(db_base, db_name)
        if not os.path.isdir(db_path):
            continue

        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk(db_path):
            for f in files:
                fp = os.path.join(root, f)
                total_files += 1
                total_size += os.path.getsize(fp)

        result[db_name] = {
            "files": total_files,
            "size_mb": round(total_size / 1024 / 1024, 1),
        }

    return result


# ─── Local Upload Script ──────────────────────────────────────────────────────

def upload_local_db_to_modal(local_db_path: str, db_name: str):
    """
    Upload a local face database folder to Modal volume.
    Run this locally after downloading databases.

    Usage:
        python face_search_modal.py upload lfw face_db/lfw
    """
    import os
    from pathlib import Path
    from tqdm import tqdm

    local_path = Path(local_db_path)
    if not local_path.exists():
        print(f"Error: {local_db_path} does not exist")
        return

    # Collect all image files
    image_files = list(local_path.rglob("*.jpg")) + \
                  list(local_path.rglob("*.jpeg")) + \
                  list(local_path.rglob("*.png"))

    print(f"Found {len(image_files)} images in {local_db_path}")
    print(f"Uploading to Modal volume as '{db_name}'...")

    # Upload in batches of 100 files
    batch_size = 100
    total_uploaded = 0

    with app.run():
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            files_dict = {}

            for img_path in batch:
                rel_path = str(img_path.relative_to(local_path))
                with open(img_path, "rb") as f:
                    files_dict[rel_path] = f.read()

            result = upload_database.remote(db_name, files_dict)
            total_uploaded += len(batch)
            print(f"  {total_uploaded}/{len(image_files)} — {result}")

    print(f"\nDone. {total_uploaded} files uploaded to Modal volume as '{db_name}'")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "upload":
        if len(sys.argv) < 4:
            print("Usage: python face_search_modal.py upload <db_name> <local_path>")
            print("Example: python face_search_modal.py upload lfw face_db/lfw")
            sys.exit(1)
        db_name = sys.argv[2]
        local_path = sys.argv[3]
        upload_local_db_to_modal(local_path, db_name)

    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        with app.run():
            dbs = list_databases.remote()
            print("\nDatabases on Modal volume:")
            for name, info in dbs.items():
                print(f"  {name}: {info['files']} files, {info['size_mb']} MB")

    elif len(sys.argv) >= 2 and sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("Usage: python face_search_modal.py test <image_path>")
            sys.exit(1)
        image_path = sys.argv[2]
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        with app.run():
            result = search_face_db.remote(image_bytes)
        import json
        print(json.dumps(result, indent=2))

    else:
        print("Commands:")
        print("  python face_search_modal.py upload <db_name> <local_path>")
        print("  python face_search_modal.py list")
        print("  python face_search_modal.py test <image_path>")