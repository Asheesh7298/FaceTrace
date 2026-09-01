# Face Identification & Blockchain Verification Pipeline

An end-to-end pipeline: face scan → multi-engine web search → blockchain-anchored tamper-evident record.

```
input.jpg  →  Face crop + multi-model embeddings
           →  Yandex / Bing / SerpApi reverse image search
           →  Engine-agreement-ranked candidate URLs
           →  SHA-256 payload hash anchored on Ethereum
           →  On-chain verification ✓
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — Face Detection & Encoding                        │
│   MTCNN detection → crop → BRISQUE quality score           │
│   GFPGAN restoration (conditional, if quality < threshold)  │
│   ArcFace (InsightFace) + FaceNet512 + VGG-Face embeddings  │
│   Weighted ensemble embedding (0.5 / 0.3 / 0.2)            │
│   EXIF metadata extraction (GPS, timestamp, edit detection) │
│   Pose angle check (frontal / angled / profile flagging)    │
└────────────────────────┬────────────────────────────────────┘
                         │ face_crop.jpg + face_hash
┌────────────────────────▼────────────────────────────────────┐
│  Stage 2 — Multi-Engine Reverse Image Search                │
│   Engine 1: Yandex (Playwright scraper — best for faces)    │
│   Engine 2: Bing Visual Search API (1000 free/month)        │
│   Engine 3: SerpApi / Google (100 free/month, last resort)  │
│   Multi-crop variants: original + flipped + sharpened       │
│   Candidate ranking: engine agreement + platform + pHash    │
│   Wayback Machine cross-reference (first-seen date)         │
└────────────────────────┬────────────────────────────────────┘
                         │ best_match URL + composite score
┌────────────────────────▼────────────────────────────────────┐
│  Stage 3 — Blockchain Anchoring & Verification              │
│   Build JSON payload (face_hash, content_hash, URL, score,  │
│     engines_agreed, platform, EXIF timestamp, pipeline ver) │
│   SHA-256 hash of payload → anchor to Ethereum              │
│   Networks: Hardhat local (demo) + Sepolia testnet (public) │
│   Verify: fetch on-chain hash → compare == local hash ✓     │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Face detection | MTCNN (facenet-pytorch) |
| Face encoding | InsightFace/ArcFace, DeepFace/FaceNet512, DeepFace/VGG-Face |
| Face restoration | GFPGAN (conditional — only on degraded inputs) |
| Image quality | BRISQUE no-reference quality metric |
| EXIF extraction | exifread |
| Reverse search | Yandex (Playwright), Bing Visual Search API, SerpApi |
| Similarity filter | imagehash (perceptual hash, Hamming distance) |
| Archive lookup | Wayback Machine CDX API (free) |
| Smart contract | Solidity 0.8.24 |
| Contract tooling | Hardhat + ethers.js |
| Blockchain client | web3.py |
| Networks | Hardhat local + Ethereum Sepolia testnet |
| GPU acceleration | CUDA via ONNX Runtime + PyTorch |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU recommended (runs on CPU but slower)
- MetaMask wallet (for Sepolia deployment only)

---

## Setup

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/your-username/face-blockchain-pipeline.git
cd face-blockchain-pipeline

pip install -r requirements.txt
python -m playwright install chromium
```

### 2. Install Node dependencies (for Hardhat)

```bash
npm install
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:

| Key | Where to get it |
|-----|----------------|
| `SERPAPI_KEY` | serpapi.com → Dashboard |
| `BING_SEARCH_KEY` | portal.azure.com → Bing Search v7 → Keys |
| `INFURA_SEPOLIA_URL` | infura.io → Project → Endpoints (Sepolia only) |
| `DEPLOYER_PRIVATE_KEY` | MetaMask → Account → Export Private Key (Sepolia only) |

### 4. Deploy the smart contract

**Local (Hardhat) — required for `--network localhost`:**
```bash
# Terminal 1: start Hardhat node
npx hardhat node

# Terminal 2: deploy
npm run deploy:local
```

**Sepolia — required for `--network sepolia`:**
```bash
npm run deploy:sepolia
```

Both commands save a `deployment_<network>.json` file that the Python pipeline reads automatically.

---

## Running the Pipeline

```bash
# Full pipeline — local Hardhat node (default)
python pipeline.py input.jpg

# Full pipeline — Sepolia testnet
python pipeline.py input.jpg --network sepolia

# Custom output directory
python pipeline.py input.jpg --output-dir results/
```

### Running individual stages

```bash
# Stage 1 only — face detection and encoding
python face/detect.py input.jpg

# Stage 2 only — reverse image search
python search/reverse_search.py output/face_crop.jpg

# Stage 3 only — anchor a result JSON
python blockchain/anchor.py output/run_<timestamp>.json localhost
```

---

## Output

Every pipeline run produces:

```
output/
├── face_crop.jpg           # Aligned face crop (224×224)
├── face_crop_restored.jpg  # GFPGAN-restored crop (if triggered)
├── face_crop_flipped.jpg   # Search variant
├── face_crop_sharpened.jpg # Search variant
├── run_<timestamp>.json    # Full structured result
└── pipeline.log            # Detailed execution log
```

### Sample `run_<timestamp>.json`

```json
{
  "input_image": "input.jpg",
  "network": "localhost",
  "face": {
    "face_crop_path": "output/face_crop.jpg",
    "face_hash": "sha256:a3f9...",
    "quality_score": 28.4,
    "pose": { "flag": "frontal", "yaw": 3.2, "confidence_multiplier": 1.0 },
    "exif": { "timestamp": "2024-03-15T14:22:00", "gps_lat": null },
    "embeddings_computed": ["ArcFace", "Facenet512", "VGG-Face"]
  },
  "search_results": {
    "engines_queried": ["yandex", "bing"],
    "engines_with_results": ["yandex", "bing"],
    "total_candidates": 12,
    "best_match": {
      "url": "https://linkedin.com/in/...",
      "platform": "linkedin.com",
      "engines": ["yandex", "bing"],
      "engine_agreement": 1.0,
      "composite_score": 0.847,
      "phash": { "accessible": true, "hamming_distance": 7, "phash_confidence": "high" },
      "wayback": { "has_archive": true, "first_seen": "2023-06-12" }
    }
  },
  "blockchain": {
    "network": "localhost",
    "tx_hash": "0x7f3a...",
    "block_number": 3,
    "gas_used": 112450,
    "contract_address": "0x5FbDB...",
    "verified": true,
    "payload": {
      "face_hash": "sha256:a3f9...",
      "content_hash": "sha256:b7d2...",
      "source_url": "https://linkedin.com/in/...",
      "ensemble_score": 0.847,
      "engines_agreed": 2,
      "pipeline_version": "1.0.0"
    }
  },
  "elapsed_seconds": 42.7,
  "pipeline_success": true
}
```

---

## Smart Contract

`FaceProof.sol` stores a proof record per payload hash:

```solidity
struct ProofRecord {
    string  payloadHash;      // SHA-256 of full JSON payload
    string  faceHash;         // SHA-256 of input face crop
    string  contentHash;      // SHA-256 of matched content
    string  sourceUrl;        // Best matched social media URL
    uint256 confidenceScore;  // Confidence × 100 (e.g. 87 = 0.87)
    uint256 timestamp;        // Block timestamp
    bool    matchFound;       // Whether a social match was found
}
```

Key functions:
- `anchor(...)` — stores a record (owner only), emits `Anchored` event
- `verify(hash)` — verifies and emits `Verified` event
- `getProof(hash)` — read-only fetch
- `totalAnchored()` — total records on this contract

---

## Blockchain Networks

| Network | Purpose | Explorer |
|---------|---------|----------|
| Hardhat local | Development & demo (always works) | N/A |
| Sepolia testnet | Public verification | [sepolia.etherscan.io](https://sepolia.etherscan.io) |

Sepolia contract address: `<filled after deployment>`

---

## Limitations

- **Social media auth walls**: Instagram, Facebook, and Twitter behind login will return the correct URL from search but the image itself cannot be downloaded for pHash comparison. The URL and metadata are still anchored on-chain.
- **Yandex scraping fragility**: Yandex has no public API. The Playwright scraper works but may be blocked by Yandex CAPTCHA on high-volume runs. Bing and SerpApi serve as automatic fallbacks.
- **Face search accuracy**: Reverse image search engines match pixel patterns, not face identity. A face that appears nowhere online (no public social presence) will return 0 results — the pipeline anchors a proof-of-scan record instead.
- **Profile/angled faces**: Faces at >45° yaw are flagged as reduced-confidence. Recognition accuracy drops for non-frontal faces.
- **SerpApi rate limit**: Free tier = 100 searches/month. With Yandex + Bing as primary engines, SerpApi is only called when both return 0 results.
- **GFPGAN model download**: First run downloads ~350MB model weights. Cached after first download.

---

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) — ArcFace model
- [DeepFace](https://github.com/serengil/deepface) — FaceNet512 + VGG-Face
- [MTCNN](https://github.com/timesler/facenet-pytorch) — Face detection
- [GFPGAN](https://github.com/TencentARC/GFPGAN) — Face restoration
- [Hardhat](https://hardhat.org) — Ethereum development
- [SerpApi](https://serpapi.com) — Google Reverse Image API
- [Wayback Machine CDX API](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server) — Archive lookup
