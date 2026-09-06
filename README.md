# FaceTrace — Face Identification & Blockchain Verification

Give it a face. It figures out **who the person is**, finds their **real social-media profile**,
**proves the identity** by matching against an independent reference photo, and anchors a
**tamper-evident record** of the whole finding on a blockchain — then re-verifies it on demand.

> **HH Goa 2026 · Task 3** — a reliable, end-to-end pipeline from a single face image to an
> immutable, self-describing on-chain proof.

```
face.jpg  ──▶  detect + encode          (MTCNN + ArcFace/FaceNet512/VGG-Face)
          ──▶  identify                 (Google Lens → "Tom Hanks")
          ──▶  find profile             (name search → instagram.com/tomhanks)
          ──▶  VERIFY BEFORE CLAIM      (DeepFace re-match vs reference photo ✓)
          ──▶  anchor SHA-256 payload   (FaceProof smart contract)
          ──▶  re-verify + tamper test  (on-chain hash still matches ✓)
```

---

## Why this design wins: *verify before you claim*

Naïve "reverse image search" is unreliable for faces — search engines return **look-alikes**, not
the actual person. In our tests Google Lens returned **60 visual matches for Tom Hanks and none of
them were Tom Hanks** (random LinkedIn head-shots that merely resemble him).

FaceTrace never trusts those. Instead it:

1. Reads the person's **name** from Google Lens's AI identity summary (`ai_overview`), extracted with
   spaCy **PERSON** named-entity recognition.
2. Runs a **name-based** search for the real profile (`"Tom Hanks" site:instagram.com …`) — highly reliable.
3. **Independently verifies** the identity: downloads an official reference photo and runs a face
   match (`DeepFace.verify`, ArcFace) against the input crop. **A match is only claimed when the face
   actually verifies.** The 60 look-alikes fail this gate and are discarded.

The result: no confident false positives, and the record we put on-chain is one we can defend.

---

## Live example (real run, `samples/tom_hanks.png`)

| Field | Value |
|-------|-------|
| Identified person | **Tom Hanks** |
| Identity verified | ✅ yes (independent reference-photo match) |
| Best profile match | `https://www.instagram.com/tomhanks/` (score 0.91) |
| Also found | x.com/tomhanks · facebook.com/TomHanks · imdb · wikipedia |
| Content hash | `sha256:59057755…` (SHA-256 of the **actual matched image bytes**) |
| On-chain TX | block #2, `verified: true` |
| Tamper test | altered record → hash **absent** from ledger → tampering detected |
| End-to-end time | **48.7s** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Face Detection & Encoding            face/detect.py        │
│    MTCNN detect + align → 224×224 crop → SHA-256 face_hash            │
│    ArcFace (InsightFace) + FaceNet512 + VGG-Face → ensemble embedding │
│    EXIF (GPS / timestamp / edit-detection) · pose (frontal/profile)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ face_crop.jpg + face_hash
┌──────────────────────────────▼──────────────────────────────────────┐
│  Stage 2 — Identity-First Search                search/               │
│    1. host crop on catbox           (Lens needs a public URL)         │
│    2. Google Lens → ai_overview → PERSON name   (identity.py, spaCy)  │
│    3. name-based social search → real profile URLs                    │
│    4. VERIFY-BEFORE-CLAIM: reference photo + DeepFace.verify          │
│    5. hash the real matched image BYTES → content_hash               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ verified identity + best_match + content_hash
┌──────────────────────────────▼──────────────────────────────────────┐
│  Stage 3 — Blockchain Anchoring & Verification  blockchain/           │
│    build self-describing JSON payload (who / how verified / how sure) │
│    SHA-256(payload) → FaceProof.anchor() on Ethereum                  │
│    read back with getProof() → confirm on-chain hash == local hash    │
│    re-verify + tamper test on demand                                  │
└─────────────────────────────────────────────────────────────────────┘
```

Every external call **degrades gracefully** — a missing key or a failed engine lowers confidence but
never crashes the run. If no identity verifies, the pipeline still anchors a **proof-of-scan** record
(`match_found: false`), so it always completes end-to-end.

---

## Tech stack

| Layer | Tool |
|-------|------|
| Face detection / alignment | MTCNN (facenet-pytorch) |
| Face encoding | InsightFace **ArcFace**, DeepFace **FaceNet512** + **VGG-Face** |
| Identity lookup | **Google Lens** via SerpApi (`ai_overview`) |
| Name extraction | **spaCy** `en_core_web_sm` PERSON NER (+ regex fallback) |
| Identity verification | `DeepFace.verify` (ArcFace) against a reference photo |
| Profile search | SerpApi Google (`site:` targeted) |
| Content fingerprint | SHA-256 of the downloaded matched-image bytes |
| Archive signal | Wayback Machine CDX API |
| Smart contract | Solidity 0.8.24 (`FaceProof.sol`) |
| Chain tooling / client | Hardhat + ethers.js · web3.py |
| Web UI | FastAPI + Server-Sent Events (live progress) |

---

## Quick start

### 0. Prerequisites
- Python 3.11 · Node.js 18+
- A **SerpApi** key (free tier, 100 searches/month) → `.env`
- ~2 GB free disk — DeepFace and InsightFace download their model weights on
  first run, so the first execution takes several minutes longer than later ones.

### 1. Install
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm     # identity NER model
npm install                                 # Hardhat + ethers
cp .env.example .env                         # then add your SERPAPI_KEY
```

### 2. Start the local blockchain (Terminal 1 — leave running)
```bash
npx hardhat node
```

### 3. Deploy the contract (Terminal 2)
```bash
npm run deploy:local        # writes deployment_localhost.json
```

### 4. Run it

**Web UI (recommended — the live demo surface):**
```bash
python app.py
# open http://localhost:8000  → pick a sample → watch the 3 stages stream live
```

**CLI:**
```bash
python pipeline.py samples/tom_hanks.png --network localhost
```

> **Note:** `npm run deploy:local` must be re-run whenever you restart `npx hardhat node`
> (the local chain resets on restart).

---

## The blockchain record

`FaceProof.sol` stores one immutable record per payload hash:

```solidity
struct ProofRecord {
    string  payloadHash;      // SHA-256 of the full JSON payload
    string  faceHash;         // SHA-256 of the input face crop
    string  contentHash;      // SHA-256 of the matched image BYTES
    string  sourceUrl;        // the matched social profile
    uint256 confidenceScore;  // verification score × 100
    uint256 timestamp;        // block timestamp
    bool    matchFound;       // whether an identity was confirmed
}
```

The hashed **payload** is self-describing — it records *who* the pipeline identified, *how it was
verified*, *how confident it was*, the source profile, and forensic metadata. So the on-chain proof
attests not merely that content existed, but exactly what FaceTrace concluded and on what basis.

### Re-verification & tamper-evidence (the "demonstrate re-verification" requirement)

`POST /api/verify-demo/<run_file>` (or the **Re-verify / Tamper test** buttons in the UI):

1. Re-reads the on-chain record for the stored hash → **verifies ✓**.
2. Alters one field, recomputes the SHA-256, and looks it up → **absent from the ledger**, proving the
   record cannot be changed without detection.

```json
{
  "genuine":  { "verified": true },
  "tampered": { "changed_field": "source_url", "found_on_chain": false }
}
```

### Networks
| Network | Purpose | Explorer |
|---------|---------|----------|
| Hardhat local | Demo — instant, pre-funded, always works | — |
| Sepolia testnet | Public, permanent proof | [sepolia.etherscan.io](https://sepolia.etherscan.io) |

Run on the public chain with `--network sepolia` (needs `INFURA_SEPOLIA_URL` + `DEPLOYER_PRIVATE_KEY`).

---

## Privacy & limitations (honest)

- **The face crop is uploaded to a public host (catbox.moe)** so Google Lens can read it — Lens
  requires an image *URL*, not a file. For public-figure demos this is harmless; for real use it is a
  privacy trade-off worth noting. Swap `search/image_host.py` for an expiring/self-hosted store if needed.
- **Identity depends on Google Lens** recognizing the person, so it works best for **findable public
  figures**. Unknown/private faces won't be identified — the pipeline then anchors a proof-of-scan record.
- **Social profile images are auth-walled**, so the content we hash is the retrievable **reference
  photo** of the identified person (the record stores both `source_url` and `matched_image_url`).
- **SerpApi free tier = 100 searches/month**; a full run uses ~3–4. Cache during development.
- **BRISQUE / GPU** are best-effort — BRISQUE falls back to a neutral score and inference runs on CPU
  if CUDA isn't available; neither blocks a successful run.

---

## Project layout

```
pipeline.py              # CLI orchestrator (3 stages, progress events)
app.py                   # FastAPI web UI (SSE live progress + tamper demo)
face/detect.py           # Stage 1 — detection + multi-model encoding
search/
  image_host.py          # catbox upload (public URL for Lens)
  identity.py            # spaCy PERSON-NER name extraction
  reverse_search.py      # Stage 2 — Lens → name → verify → content hash
blockchain/
  anchor.py              # Stage 3 — web3.py anchor + verify
  contracts/FaceProof.sol
  scripts/{deploy,verify}.js
templates/index.html     # web frontend
samples/                 # demo face images
```

---

## Credits

[InsightFace](https://github.com/deepinsight/insightface) ·
[DeepFace](https://github.com/serengil/deepface) ·
[facenet-pytorch](https://github.com/timesler/facenet-pytorch) ·
[spaCy](https://spacy.io) ·
[SerpApi](https://serpapi.com) ·
[Hardhat](https://hardhat.org) ·
[Wayback Machine](https://archive.org/web/)
