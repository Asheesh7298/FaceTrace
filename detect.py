"""
face/detect.py
Multi-model face detection and encoding.

Models used:
  - MTCNN          : detection + alignment
  - InsightFace    : ArcFace 512-d embedding (CUDA)
  - DeepFace       : FaceNet512 + VGG-Face embeddings (CUDA)

Quality pipeline:
  - BRISQUE score  : decides whether to sharpen before encoding
  - GFPGAN         : conditional restoration for badly degraded crops
  - Pose check     : flags profile/angled faces, adjusts confidence
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import exifread

logger = logging.getLogger(__name__)


# ─── BRISQUE Quality Score ───────────────────────────────────────────────────

def brisque_score(image_path: str) -> float:
    """
    Returns a BRISQUE no-reference quality score.
    Lower = better quality. Range ~0-100.
    Falls back to 50.0 if library unavailable.
    """
    try:
        from brisque import BRISQUE
        brisq = BRISQUE(url=False)
        img = cv2.imread(image_path)
        score = brisq.score(img)
        return float(max(0.0, min(100.0, score)))
    except Exception as e:
        logger.warning(f"BRISQUE unavailable ({e}), defaulting to 50.0")
        return 50.0


# ─── GFPGAN Face Restoration ─────────────────────────────────────────────────

def restore_face(image_path: str, output_path: str) -> str:
    """
    Runs GFPGAN face restoration on a degraded crop.
    Returns output_path on success, original image_path on failure.
    """
    try:
        import torch
        from gfpgan import GFPGANer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            device=device,
        )

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        _, _, restored = restorer.enhance(
            img, has_aligned=False, only_center_face=True, paste_back=True
        )

        if restored is not None:
            cv2.imwrite(output_path, restored[0])
            logger.info(f"GFPGAN restoration saved to {output_path}")
            return output_path

    except Exception as e:
        logger.warning(f"GFPGAN restoration failed ({e}), using original")

    return image_path


# ─── EXIF Extraction ─────────────────────────────────────────────────────────

def extract_exif(image_path: str) -> dict:
    """
    Extracts useful EXIF metadata from the input image.
    Returns dict with gps, timestamp, camera, software fields.
    """
    exif_data = {
        "gps_lat": None,
        "gps_lon": None,
        "timestamp": None,
        "camera_model": None,
        "software": None,
        "was_edited": False,
    }

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSLongitude", details=False)

        if "Image DateTime" in tags:
            exif_data["timestamp"] = str(tags["Image DateTime"])

        if "Image Model" in tags:
            exif_data["camera_model"] = str(tags["Image Model"])

        if "Image Software" in tags:
            sw = str(tags["Image Software"])
            exif_data["software"] = sw
            edit_keywords = ["photoshop", "lightroom", "snapseed", "facetune", "vsco"]
            exif_data["was_edited"] = any(k in sw.lower() for k in edit_keywords)

        def dms_to_decimal(dms_tag, ref_tag):
            try:
                dms = [float(x.num) / float(x.den) for x in dms_tag.values]
                decimal = dms[0] + dms[1] / 60 + dms[2] / 3600
                if str(ref_tag) in ["S", "W"]:
                    decimal = -decimal
                return round(decimal, 6)
            except Exception:
                return None

        if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
            exif_data["gps_lat"] = dms_to_decimal(
                tags["GPS GPSLatitude"], tags["GPS GPSLatitudeRef"]
            )
        if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
            exif_data["gps_lon"] = dms_to_decimal(
                tags["GPS GPSLongitude"], tags["GPS GPSLongitudeRef"]
            )

    except Exception as e:
        logger.warning(f"EXIF extraction failed: {e}")

    return exif_data


# ─── Pose Estimation ─────────────────────────────────────────────────────────

def estimate_pose(face_landmarks: dict) -> dict:
    """
    Estimates yaw/pitch from InsightFace landmark positions.
    Returns pose dict with angles and a confidence multiplier.
    """
    pose = {"yaw": 0.0, "pitch": 0.0, "confidence_multiplier": 1.0, "flag": "frontal"}

    try:
        # InsightFace gives us 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
        left_eye  = np.array(face_landmarks["left_eye"])
        right_eye = np.array(face_landmarks["right_eye"])
        nose      = np.array(face_landmarks["nose"])

        # Yaw: horizontal asymmetry between eyes and nose
        eye_center = (left_eye + right_eye) / 2
        yaw = float(nose[0] - eye_center[0])

        # Pitch: vertical offset
        pitch = float(nose[1] - eye_center[1])

        pose["yaw"]   = round(yaw, 2)
        pose["pitch"] = round(pitch, 2)

        abs_yaw = abs(yaw)
        if abs_yaw < 15:
            pose["flag"] = "frontal"
            pose["confidence_multiplier"] = 1.0
        elif abs_yaw < 45:
            pose["flag"] = "angled"
            pose["confidence_multiplier"] = 0.85
            logger.warning(f"Face is angled ({yaw:.1f}°) — confidence reduced by 15%")
        else:
            pose["flag"] = "profile"
            pose["confidence_multiplier"] = 0.65
            logger.warning(f"Face is near-profile ({yaw:.1f}°) — confidence reduced by 35%")

    except Exception as e:
        logger.warning(f"Pose estimation failed: {e}")

    return pose


# ─── Multi-Model Embedding ────────────────────────────────────────────────────

def get_insightface_embedding(image_path: str) -> Optional[np.ndarray]:
    """ArcFace 512-d embedding via InsightFace."""
    try:
        import insightface
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        img = cv2.imread(image_path)
        faces = app.get(img)

        if not faces:
            logger.warning("InsightFace: no face detected")
            return None

        # Pick largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding  # 512-d normalized

    except Exception as e:
        logger.warning(f"InsightFace embedding failed: {e}")
        return None


def get_deepface_embeddings(image_path: str) -> dict:
    """
    FaceNet512 + VGG-Face embeddings via DeepFace.
    Returns dict of model_name → embedding array.
    """
    embeddings = {}
    models = ["Facenet512", "VGG-Face"]

    for model in models:
        try:
            from deepface import DeepFace

            result = DeepFace.represent(
                img_path=image_path,
                model_name=model,
                detector_backend="mtcnn",
                enforce_detection=False,
            )

            if result:
                embeddings[model] = np.array(result[0]["embedding"])
                logger.info(f"DeepFace/{model}: embedding shape {embeddings[model].shape}")

        except Exception as e:
            logger.warning(f"DeepFace/{model} failed: {e}")

    return embeddings


# ─── Main Detection Function ──────────────────────────────────────────────────

def detect_and_encode(
    image_path: str,
    output_dir: str = ".",
    force_restore: bool = False,
) -> dict:
    """
    Full face detection and encoding pipeline.

    Returns a result dict with:
      - face_crop_path   : path to aligned face crop
      - face_hash        : SHA-256 of the face crop
      - quality_score    : BRISQUE score (lower = better)
      - pose             : yaw/pitch/flag/confidence_multiplier
      - exif             : extracted EXIF metadata
      - embeddings       : dict of model → np.ndarray
      - ensemble_embedding : weighted average embedding (normalized)
      - success          : bool
      - error            : error message if failed
    """
    result = {
        "input_image": image_path,
        "face_crop_path": None,
        "face_hash": None,
        "quality_score": None,
        "pose": {},
        "exif": {},
        "embeddings": {},
        "ensemble_embedding": None,
        "success": False,
        "error": None,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. EXIF extraction ────────────────────────────────────────────────────
    logger.info("Extracting EXIF metadata...")
    result["exif"] = extract_exif(image_path)
    if result["exif"]["gps_lat"]:
        logger.info(f"GPS found: {result['exif']['gps_lat']}, {result['exif']['gps_lon']}")
    if result["exif"]["was_edited"]:
        logger.warning(f"Image appears edited (software: {result['exif']['software']})")

    # ── 2. Quality scoring ────────────────────────────────────────────────────
    logger.info("Computing BRISQUE quality score...")
    quality_score = brisque_score(image_path)
    result["quality_score"] = quality_score
    logger.info(f"BRISQUE score: {quality_score:.1f} (lower = better)")

    # ── 3. MTCNN detection + crop ─────────────────────────────────────────────
    logger.info("Running MTCNN face detection...")
    try:
        import torch
        from facenet_pytorch import MTCNN
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        mtcnn = MTCNN(
            image_size=224,
            margin=20,
            keep_all=False,
            device=device,
            post_process=False,
        )

        pil_img = PILImage.open(image_path).convert("RGB")
        boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or len(boxes) == 0:
            result["error"] = "No face detected by MTCNN"
            logger.error(result["error"])
            return result

        # Pick highest confidence detection
        best_idx = int(np.argmax(probs))
        box = boxes[best_idx]
        landmark = landmarks[best_idx]

        # Crop with padding
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = 20
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(pil_img.width, x2 + pad), min(pil_img.height, y2 + pad)

        face_crop = pil_img.crop((x1, y1, x2, y2)).resize((224, 224))
        crop_path = str(output_dir / "face_crop.jpg")
        face_crop.save(crop_path, quality=95)
        logger.info(f"Face crop saved: {crop_path} (detection confidence: {probs[best_idx]:.3f})")

        # Pose from landmarks
        lm = landmark
        pose_input = {
            "left_eye":    lm[0].tolist(),
            "right_eye":   lm[1].tolist(),
            "nose":        lm[2].tolist(),
            "left_mouth":  lm[3].tolist(),
            "right_mouth": lm[4].tolist(),
        }
        result["pose"] = estimate_pose(pose_input)

    except Exception as e:
        result["error"] = f"MTCNN detection failed: {e}"
        logger.error(result["error"])
        return result

    # ── 4. Conditional enhancement ────────────────────────────────────────────
    working_crop = crop_path
    crop_size = face_crop.size[0]  # width after resize = 224

    if force_restore or quality_score > 60 or crop_size < 112:
        logger.info(f"Quality score {quality_score:.1f} > 60 or small crop — applying GFPGAN...")
        restored_path = str(output_dir / "face_crop_restored.jpg")
        working_crop = restore_face(crop_path, restored_path)
    else:
        logger.info(f"Quality score {quality_score:.1f} — no restoration needed")

    result["face_crop_path"] = working_crop

    # ── 5. Face hash ──────────────────────────────────────────────────────────
    with open(working_crop, "rb") as f:
        result["face_hash"] = "sha256:" + hashlib.sha256(f.read()).hexdigest()
    logger.info(f"Face hash: {result['face_hash']}")

    # ── 6. Multi-model embedding ──────────────────────────────────────────────
    logger.info("Computing multi-model embeddings...")

    # InsightFace / ArcFace (weight: 0.5 — most accurate)
    arcface_emb = get_insightface_embedding(working_crop)
    if arcface_emb is not None:
        result["embeddings"]["ArcFace"] = arcface_emb

    # DeepFace models (weight: 0.3 + 0.2)
    deepface_embs = get_deepface_embeddings(working_crop)
    result["embeddings"].update(deepface_embs)

    logger.info(f"Embeddings computed: {list(result['embeddings'].keys())}")

    # ── 7. Weighted ensemble embedding ────────────────────────────────────────
    model_weights = {
        "ArcFace":    0.50,
        "Facenet512": 0.30,
        "VGG-Face":   0.20,
    }

    weighted_sum = None
    total_weight = 0.0

    for model_name, weight in model_weights.items():
        if model_name in result["embeddings"]:
            emb = result["embeddings"][model_name].astype(np.float64)
            # Normalize to unit vector before combining
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            if weighted_sum is None:
                weighted_sum = weight * emb
            else:
                # Resize if dimensions differ
                if emb.shape != weighted_sum.shape:
                    emb = emb[: weighted_sum.shape[0]]
                weighted_sum += weight * emb
            total_weight += weight

    if weighted_sum is not None and total_weight > 0:
        ensemble = weighted_sum / total_weight
        norm = np.linalg.norm(ensemble)
        if norm > 0:
            ensemble = ensemble / norm
        result["ensemble_embedding"] = ensemble
        logger.info(f"Ensemble embedding: shape {ensemble.shape}, norm {np.linalg.norm(ensemble):.4f}")

    result["success"] = True
    logger.info("Face detection and encoding complete ✓")
    return result


if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python face/detect.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect_and_encode(image_path, output_dir="output")

    # Don't print raw numpy arrays to stdout
    printable = {k: v for k, v in result.items() if k != "embeddings" and k != "ensemble_embedding"}
    printable["embeddings_computed"] = list(result["embeddings"].keys())
    print(json.dumps(printable, indent=2, default=str))
