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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)
logger  = logging.getLogger("pipeline")


def setup_logging(output_dir: str):
    import io
    import threading
    log_path = Path(output_dir) / "pipeline.log"

    handlers = [logging.FileHandler(log_path, encoding="utf-8")]

    # Only add stdout handler if we're in the main thread (not the web server's bg thread)
    if threading.current_thread() is threading.main_thread():
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") if hasattr(sys.stdout, "buffer") else sys.stdout
        handlers.append(logging.StreamHandler(stream))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
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
    table.add_row("Engines queried",  ", ".join(search.get("engines_queried", [])))
    table.add_row("Engines with hits",", ".join(search.get("engines_with_results", [])))
    table.add_row("Total candidates", str(search.get("total_candidates", 0)))
    table.add_row("Best match URL",   best.get("url", "none")[:80] if best else "none")
    table.add_row("Platform",         best.get("platform", "-"))
    table.add_row("Composite score",  str(best.get("composite_score", "-")))
    table.add_row("Engine agreement", str(best.get("engine_agreement", "-")))
    table.add_row("Wayback first seen", best.get("wayback", {}).get("first_seen") or "-")
    table.add_row("Blockchain network", chain.get("network", "-"))
    table.add_row("TX hash",          chain.get("tx_hash", "-"))
    table.add_row("Block number",     str(chain.get("block_number", "-")))
    table.add_row("Verified ✓",       "✅ YES" if chain.get("verified") else "❌ NO")
    table.add_row("Etherscan",        chain.get("etherscan_url") or "N/A (local)")
    table.add_row("Elapsed seconds",  str(result.get("elapsed_seconds", "-")))

    console.print(table)


def run_pipeline(image_path: str, network: str = "localhost", output_dir: str = "output", progress_callback=None) -> dict:
    def emit(event_type, data):
        if progress_callback:
            try:
                progress_callback(event_type, data)
            except Exception:
                pass

    start_time = time.time()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    print_banner()
    logger.info(f"Input: {image_path} | Network: {network} | Output: {output_dir}")

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
    emit("stage", {"stage": 1, "status": "running", "message": "Detecting face and computing embeddings..."})

    from face.detect import detect_and_encode

    face_result = detect_and_encode(image_path, output_dir=output_dir)

    face_summary = {k: v for k, v in face_result.items()
                    if k not in ("embeddings", "ensemble_embedding")}
    face_summary["embeddings_computed"] = list(face_result.get("embeddings", {}).keys())
    result["face"] = face_summary

    if not face_result["success"]:
        console.print(f"[bold red]Stage 1 FAILED: {face_result['error']}[/bold red]")
        emit("stage", {"stage": 1, "status": "error", "message": face_result["error"]})
        result["elapsed_seconds"] = round(time.time() - start_time, 1)
        emit("complete", {"success": False, "elapsed": result["elapsed_seconds"], "error": face_result["error"]})
        _save_result(result, output_dir)
        return result

    emit("face", {
        "face_hash": face_result["face_hash"],
        "quality": face_result["quality_score"],
        "pose": face_result.get("pose", {}).get("flag", "unknown"),
        "pose_yaw": face_result.get("pose", {}).get("yaw", 0),
        "pose_pitch": face_result.get("pose", {}).get("pitch", 0),
        "crop_url": "/output/face_crop.jpg",
        "embeddings": face_summary["embeddings_computed"],
        "exif": face_result.get("exif", {}),
    })
    emit("stage", {"stage": 1, "status": "done", "message": f"Face detected — {len(face_summary['embeddings_computed'])} models"})

    console.print(f"[green]Stage 1 complete ✓[/green] — "
                  f"Face crop: {face_result['face_crop_path']} | "
                  f"Models: {face_summary['embeddings_computed']} | "
                  f"Quality: {face_result['quality_score']:.1f}")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Multi-Engine Reverse Image & Face Search
    # ════════════════════════════════════════════════════════════════════════
    print_stage(2, "Face & Reverse Image Search (FaceCheck → Yandex → Bing → SerpApi)")
    emit("stage", {"stage": 2, "status": "running", "message": "Searching web & social networks for matching faces..."})

    from search.reverse_search import multi_engine_search

    search_result = multi_engine_search(
        face_crop_path=face_result["face_crop_path"],
        output_dir=output_dir,
        exif_data=face_result.get("exif"),
    )
    result["search_results"] = search_result

    if search_result["success"]:
        best = search_result["best_match"]
        emit("match", {
            "url": best.get("url", ""),
            "platform": best.get("platform", "unknown"),
            "score": best.get("composite_score", 0),
            "engines": best.get("engines", []),
            "total_candidates": search_result.get("total_candidates", 0),
        })
        emit("stage", {"stage": 2, "status": "done", "message": f"Found {search_result['total_candidates']} candidates — best: {best.get('platform', 'unknown')}"})
        console.print(f"[green]Stage 2 complete ✓[/green] — "
                      f"Found {search_result['total_candidates']} candidates | "
                      f"Best: {best['url'][:60]}... | "
                      f"Score: {best['composite_score']}")
    else:
        emit("stage", {"stage": 2, "status": "done", "message": "No web matches found — anchoring proof-of-scan"})
        console.print("[yellow]Stage 2: No social match found — "
                      "will anchor proof-of-scan record[/yellow]")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Blockchain Anchoring & Verification
    # ════════════════════════════════════════════════════════════════════════
    print_stage(3, f"Blockchain Anchoring ({network})")
    emit("stage", {"stage": 3, "status": "running", "message": f"Anchoring to {network}..."})

    from blockchain.anchor import BlockchainAnchor, build_payload, sha256_of_str
    import json as _json

    best = search_result.get("best_match") or {}
    match_found = search_result.get("success", False)

    content_hash = (
        sha256_of_str(best["url"]) if match_found else "none"
    )

    payload = build_payload(
        face_hash          = face_result["face_hash"],
        content_hash       = content_hash,
        source_url         = best.get("url", "none"),
        ensemble_score     = best.get("composite_score", 0.0),
        engines_agreed     = len(best.get("engines", [])),
        platform           = best.get("platform", "unknown"),
        exif_timestamp     = face_result.get("exif", {}).get("timestamp"),
        wayback_first_seen = best.get("wayback", {}).get("first_seen"),
        match_found        = match_found,
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

        emit("blockchain", {
            "tx_hash": anchor_result.get("tx_hash", ""),
            "block_number": anchor_result.get("block_number", 0),
            "gas_used": anchor_result.get("gas_used", 0),
            "contract_address": anchor_result.get("contract_address", ""),
            "payload_hash": payload_hash,
            "verified": verify_result["verified"],
            "network": network,
            "etherscan_url": anchor_result.get("etherscan_url"),
        })

        if verify_result["verified"]:
            emit("stage", {"stage": 3, "status": "done", "message": f"Verified on block {anchor_result['block_number']}"})
            console.print(f"[green]Stage 3 complete ✓[/green] — "
                          f"TX: {anchor_result['tx_hash']} | "
                          f"Block: {anchor_result['block_number']} | "
                          f"Verified: ✅")
        else:
            emit("stage", {"stage": 3, "status": "error", "message": "Anchored but verification failed"})
            console.print("[red]Stage 3: Anchoring succeeded but verification failed[/red]")

    except Exception as e:
        logger.error(f"Blockchain stage failed: {e}")
        console.print(f"[red]Stage 3 FAILED: {e}[/red]")
        result["blockchain"]["error"] = str(e)
        emit("stage", {"stage": 3, "status": "error", "message": str(e)})

    # ════════════════════════════════════════════════════════════════════════
    # Final summary
    # ════════════════════════════════════════════════════════════════════════
    result["elapsed_seconds"] = round(time.time() - start_time, 1)
    result["pipeline_success"] = (
        face_result["success"]
        and result["blockchain"].get("verified", False)
    )

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
    emit("complete", {"success": result["pipeline_success"], "elapsed": result["elapsed_seconds"]})
    return result


def _save_result(result: dict, output_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"run_{ts}.json"
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
