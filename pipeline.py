"""
pipeline.py — Main orchestrator
================================
Usage:
    python pipeline.py <image_path> [--network localhost|sepolia] [--output-dir output]

Runs all 3 stages end-to-end:
    1. Face detection & multi-model encoding
    2. Multi-engine reverse image search & ranking
    3. Blockchain anchoring & verification

Outputs:
    - output/face_crop.jpg          : aligned face crop
    - output/run_<timestamp>.json   : full structured result
    - Console: live progress + final summary
"""

import os
import sys

# Fix Windows asyncio + Playwright conflict, and force UTF-8 output so the
# rich banner (arrows / box-drawing) doesn't crash on a cp1252 console.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

load_dotenv()

console = Console()
logger  = logging.getLogger("pipeline")


def setup_logging(output_dir: str):
    log_path = Path(output_dir) / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Face Identification & Blockchain Verification Pipeline[/bold cyan]\n"
        "[dim]Face scan → Web search → Blockchain anchor → Verify[/dim]",
        border_style="cyan",
    ))


def print_stage(n: int, title: str):
    console.print(f"\n[bold yellow]{'━' * 55}[/bold yellow]")
    console.print(f"[bold yellow]  Stage {n}/3 — {title}[/bold yellow]")
    console.print(f"[bold yellow]{'━' * 55}[/bold yellow]")


def print_result_table(result: dict):
    table = Table(title="Pipeline Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    face = result.get("face", {})
    search = result.get("search_results", {})
    chain = result.get("blockchain", {})
    best = search.get("best_match") or {}

    table.add_row("Input image",      result.get("input_image", "-"))
    table.add_row("Face detected",    "✅ Yes" if face.get("success") else "❌ No")
    table.add_row("Quality (BRISQUE)",str(face.get("quality_score", "-")))
    table.add_row("Pose",             face.get("pose", {}).get("flag", "-"))
    table.add_row("Embeddings",       ", ".join(face.get("embeddings_computed", [])))
    if search.get("person_name"):
        identity_str = search["person_name"]
    elif search.get("lens_suggested_name"):
        identity_str = f"unconfirmed (Lens guessed: {search['lens_suggested_name']})"
    else:
        identity_str = "unidentified"
    table.add_row("Identified person", identity_str)
    table.add_row("Identity verified", "✅ Yes" if search.get("name_verified") else "❌ No")
    table.add_row("Verification", f"{search.get('verification_matches', 0)}/"
                                   f"{search.get('verification_refs', 0)} refs · "
                                   f"score {search.get('verification_score', '-')}")
    table.add_row("Engines with hits",", ".join(search.get("engines_with_results", [])))
    table.add_row("Total candidates", str(search.get("total_candidates", 0)))
    table.add_row("Best match URL",   (best.get("url", "none") or "none")[:80] if best else "none")
    table.add_row("Platform",         best.get("platform", "-"))
    table.add_row("Composite score",  str(best.get("composite_score", "-")))
    table.add_row("Content hash",     (search.get("matched_content") or {}).get("content_hash", "-"))
    table.add_row("Wayback first seen", best.get("wayback", {}).get("first_seen") or "-")
    table.add_row("Blockchain network", chain.get("network", "-"))
    table.add_row("TX hash",          chain.get("tx_hash", "-"))
    table.add_row("Block number",     str(chain.get("block_number", "-")))
    table.add_row("Verified ✓",       "✅ YES" if chain.get("verified") else "❌ NO")
    table.add_row("Etherscan",        chain.get("etherscan_url") or "N/A (local)")
    table.add_row("Elapsed seconds",  str(result.get("elapsed_seconds", "-")))

    console.print(table)


def run_pipeline(image_path: str, network: str = "localhost", output_dir: str = "output",
                 progress_callback=None) -> dict:
    start_time = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    def emit(event: str, data: dict):
        """Push a structured event to the web UI (if attached) and log it."""
        if progress_callback:
            try:
                progress_callback(event, data)
            except Exception as e:
                logger.warning(f"progress_callback error: {e}")

    print_banner()
    logger.info(f"Input: {image_path} | Network: {network} | Output: {output_dir}")
    emit("pipeline_start", {"input_image": image_path, "network": network})

    result = {
        "input_image":     image_path,
        "network":         network,
        "started_at":      datetime.now(timezone.utc).isoformat(),
        "face":            {},
        "search_results":  {},
        "blockchain":      {},
        "elapsed_seconds": None,
        "pipeline_success": False,
    }

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Face Detection & Encoding
    # ════════════════════════════════════════════════════════════════════════
    print_stage(1, "Face Detection & Multi-Model Encoding")
    emit("stage_start", {"stage": 1, "title": "Face Detection & Encoding"})

    from face.detect import detect_and_encode

    face_result = detect_and_encode(image_path, output_dir=output_dir)

    face_summary = {k: v for k, v in face_result.items()
                    if k not in ("embeddings", "ensemble_embedding")}
    face_summary["embeddings_computed"] = list(face_result.get("embeddings", {}).keys())
    result["face"] = face_summary

    if not face_result["success"]:
        console.print(f"[bold red]Stage 1 FAILED: {face_result['error']}[/bold red]")
        emit("stage_error", {"stage": 1, "error": face_result["error"]})
        result["elapsed_seconds"] = round(time.time() - start_time, 1)
        _save_result(result, output_dir)
        return result

    console.print(f"[green]Stage 1 complete ✓[/green] — "
                  f"Face crop: {face_result['face_crop_path']} | "
                  f"Models: {face_summary['embeddings_computed']} | "
                  f"Quality: {face_result['quality_score']:.1f}")
    emit("stage_complete", {
        "stage": 1,
        "face_crop": face_result["face_crop_path"],
        "face_hash": face_result["face_hash"],
        "quality_score": face_result["quality_score"],
        "pose": face_result.get("pose", {}).get("flag"),
        "embeddings": face_summary["embeddings_computed"],
    })

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Multi-Engine Reverse Image Search
    # ════════════════════════════════════════════════════════════════════════
    print_stage(2, "Identity Search (Lens → Name → Verify)")
    emit("stage_start", {"stage": 2, "title": "Identity & Social Search"})

    from search.reverse_search import multi_engine_search

    search_result = multi_engine_search(
        face_crop_path=face_result["face_crop_path"],
        output_dir=output_dir,
        exif_data=face_result.get("exif"),
        progress=emit,
        identify_image_path=image_path,   # original photo → best Lens recognition
    )
    result["search_results"] = search_result

    if search_result["success"]:
        best = search_result["best_match"]
        console.print(f"[green]Stage 2 complete ✓[/green] — "
                      f"Identified: {search_result['person_name']} "
                      f"(verified {search_result['verification_matches']}/"
                      f"{search_result['verification_refs']} refs) | "
                      f"Best: {best['url'][:60]} | "
                      f"Score: {best['composite_score']}")
    else:
        suggested = search_result.get("lens_suggested_name")
        if suggested:
            console.print(f"[yellow]Stage 2: Lens suggested '{suggested}' but it did NOT "
                          f"verify ({search_result['verification_matches']}/"
                          f"{search_result['verification_refs']} refs) — not claimed. "
                          f"Anchoring proof-of-scan record[/yellow]")
        else:
            console.print("[yellow]Stage 2: No identity confirmed — "
                          "anchoring proof-of-scan record[/yellow]")
    emit("stage_complete", {
        "stage": 2,
        "person_name": search_result.get("person_name"),
        "lens_suggested_name": search_result.get("lens_suggested_name"),
        "name_verified": search_result.get("name_verified"),
        "verification_score": search_result.get("verification_score"),
        "verification_matches": search_result.get("verification_matches"),
        "verification_refs": search_result.get("verification_refs"),
        "verification_photo_url": search_result.get("verification_photo_url"),
        "best_match": search_result.get("best_match"),
        "total_candidates": search_result.get("total_candidates"),
        "match_found": search_result["success"],
    })

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Blockchain Anchoring & Verification
    # ════════════════════════════════════════════════════════════════════════
    print_stage(3, f"Blockchain Anchoring ({network})")
    emit("stage_start", {"stage": 3, "title": f"Blockchain Anchoring ({network})"})

    from blockchain.anchor import BlockchainAnchor, build_payload, sha256_of_str
    import json as _json

    best = search_result.get("best_match") or {}
    match_found = search_result.get("success", False)
    matched_content = search_result.get("matched_content") or {}

    # Real content fingerprint: SHA-256 of the actual matched image BYTES
    # (downloaded during verification). Falls back to "none" when nothing
    # retrievable was matched — the record is still anchored (proof-of-scan).
    content_hash = matched_content.get("content_hash", "none")

    payload = build_payload(
        face_hash          = face_result["face_hash"],
        content_hash       = content_hash,
        source_url         = best.get("url", "none"),
        ensemble_score     = search_result.get("verification_score", 0.0),
        engines_agreed     = len(search_result.get("engines_with_results", [])),
        platform           = best.get("platform", "unknown"),
        exif_timestamp     = face_result.get("exif", {}).get("timestamp"),
        wayback_first_seen = best.get("wayback", {}).get("first_seen"),
        match_found        = match_found,
        person_name        = search_result.get("person_name"),
        identity_verified  = search_result.get("name_verified", False),
        verification_score = search_result.get("verification_score", 0.0),
        matched_image_url  = matched_content.get("image_url"),
    )

    payload_json = _json.dumps(payload, sort_keys=True)
    payload_hash = sha256_of_str(payload_json)

    try:
        anchor_client = BlockchainAnchor(network=network)

        console.print(f"Anchoring to [cyan]{network}[/cyan]...")
        anchor_result = anchor_client.anchor(payload)

        console.print(f"Verifying on-chain record...")
        verify_result = anchor_client.verify(anchor_result["payload_hash"])

        result["blockchain"] = {
            **anchor_result,
            "verified":        verify_result["verified"],
            "on_chain_record": verify_result,
            "payload":         payload,
            "payload_hash":    payload_hash,
        }

        if verify_result["verified"]:
            console.print(f"[green]Stage 3 complete ✓[/green] — "
                          f"TX: {anchor_result['tx_hash']} | "
                          f"Block: {anchor_result['block_number']} | "
                          f"Verified: ✅")
        else:
            console.print("[red]Stage 3: Anchoring succeeded but verification failed[/red]")

        emit("stage_complete", {
            "stage": 3,
            "tx_hash": anchor_result["tx_hash"],
            "block_number": anchor_result["block_number"],
            "contract_address": anchor_result["contract_address"],
            "payload_hash": anchor_result["payload_hash"],
            "verified": verify_result["verified"],
            "etherscan_url": anchor_result.get("etherscan_url"),
            "payload": payload,
        })

    except Exception as e:
        logger.error(f"Blockchain stage failed: {e}")
        console.print(f"[red]Stage 3 FAILED: {e}[/red]")
        emit("stage_error", {"stage": 3, "error": str(e)})
        result["blockchain"]["error"] = str(e)

    # ════════════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════════════
    result["elapsed_seconds"] = round(time.time() - start_time, 1)
    result["pipeline_success"] = (
        face_result["success"]
        and result["blockchain"].get("verified", False)
    )
    emit("pipeline_complete", {
        "pipeline_success": result["pipeline_success"],
        "elapsed_seconds": result["elapsed_seconds"],
    })

    console.print()
    print_result_table(result)

    if result["pipeline_success"]:
        console.print(Panel.fit(
            "[bold green]✅ Pipeline completed successfully end-to-end[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Pipeline completed with partial results[/bold yellow]",
            border_style="yellow",
        ))

    _save_result(result, output_dir)
    return result


def _save_result(result: dict, output_dir: str):
    import shutil
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = Path(output_dir) / f"run_{ts}.json"
    result["result_file"] = out_path.name

    # Snapshot the face crop under a UNIQUE per-run name. The pipeline reuses
    # output/face_crop.jpg every scan, so without this the browser would cache
    # (and history would reuse) a stale crop from a previous run.
    try:
        fc = (result.get("face") or {}).get("face_crop_path")
        if fc and Path(fc).exists():
            unique = Path(output_dir) / f"crop_{ts}.jpg"
            shutil.copyfile(fc, unique)
            result["face"]["face_crop_display"] = unique.name
        rp = Path(output_dir) / "reference_photo.jpg"
        if rp.exists() and result.get("search_results"):
            uref = Path(output_dir) / f"ref_{ts}.jpg"
            shutil.copyfile(rp, uref)
            result["search_results"]["reference_photo_display"] = uref.name
    except Exception as e:
        logger.warning(f"crop snapshot failed: {e}")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    console.print(f"\n[dim]Full result saved to: {out_path}[/dim]")


def main():
    parser = argparse.ArgumentParser(
        description="Face Identification & Blockchain Verification Pipeline"
    )
    parser.add_argument("image", help="Path to input face image")
    parser.add_argument(
        "--network",
        choices=["localhost", "sepolia"],
        default="localhost",
        help="Blockchain network (default: localhost)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for outputs (default: output/)",
    )

    args = parser.parse_args()

    if not Path(args.image).exists():
        console.print(f"[red]Error: Image not found: {args.image}[/red]")
        sys.exit(1)

    result = run_pipeline(
        image_path=args.image,
        network=args.network,
        output_dir=args.output_dir,
    )

    sys.exit(0 if result["pipeline_success"] else 1)


if __name__ == "__main__":
    main()