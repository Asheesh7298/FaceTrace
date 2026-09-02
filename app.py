"""
app.py — FaceTrace Web Server
================================
FastAPI server with SSE streaming for the FaceTrace pipeline.

Usage:
    python app.py

Endpoints:
    GET  /              → Frontend UI
    POST /api/scan      → Start a new scan
    GET  /api/stream/{id} → SSE stream of pipeline progress
    GET  /api/history    → List past scans
    GET  /api/result/{f} → Get specific run result
    GET  /api/samples    → List available sample images
    GET  /docs           → Auto-generated API docs
"""

import os
import sys
import json
import uuid
import shutil
import asyncio
import threading
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

load_dotenv()

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="FaceTrace", description="Face Identification & Blockchain Verification Pipeline")

# Static file mounts
Path("output").mkdir(exist_ok=True)
Path("samples").mkdir(exist_ok=True)
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

# In-memory scan state
scans: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/scan")
async def start_scan(
    file: UploadFile = File(None),
    sample: str = Form(None),
    network: str = Form("localhost"),
):
    scan_id = str(uuid.uuid4())[:8]

    if sample:
        image_path = str(Path("samples") / sample)
        if not Path(image_path).exists():
            raise HTTPException(status_code=404, detail=f"Sample not found: {sample}")
    elif file:
        upload_dir = Path("output") / "uploads"
        upload_dir.mkdir(exist_ok=True)
        image_path = str(upload_dir / f"{scan_id}_{file.filename}")
        with open(image_path, "wb") as f:
            content = await file.read()
            f.write(content)
    else:
        raise HTTPException(status_code=400, detail="No file or sample provided")

    scans[scan_id] = {
        "id": scan_id,
        "image_path": image_path,
        "network": network,
        "status": "pending",
        "events": [],
        "result": None,
        "done": threading.Event(),
        "new_event": threading.Event(),
    }

    thread = threading.Thread(target=_run_scan, args=(scan_id,), daemon=True)
    thread.start()

    return {"scan_id": scan_id, "image_path": image_path, "network": network}


def _run_scan(scan_id: str):
    """Runs the pipeline in a background thread, pushing events."""
    scan = scans[scan_id]
    scan["status"] = "running"

    def progress_callback(event_type: str, data: dict):
        scan["events"].append({"event": event_type, "data": data})
        scan["new_event"].set()  # wake up SSE generator

    try:
        # Isolate pipeline logging to file-only so it doesn't corrupt
        # uvicorn's stdout/stderr handlers
        import logging as _logging
        pipeline_logger = _logging.getLogger("pipeline")
        pipeline_logger.handlers.clear()
        pipeline_logger.propagate = False
        fh = _logging.FileHandler("output/pipeline.log", encoding="utf-8")
        fh.setFormatter(_logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        pipeline_logger.addHandler(fh)
        pipeline_logger.setLevel(_logging.INFO)

        from pipeline import run_pipeline
        result = run_pipeline(
            image_path=scan["image_path"],
            network=scan["network"],
            output_dir="output",
            progress_callback=progress_callback,
        )
        scan["result"] = result
        scan["status"] = "done"
    except Exception as e:
        scan["events"].append({"event": "error", "data": {"message": str(e)}})
        scan["status"] = "error"
    finally:
        scan["new_event"].set()
        scan["done"].set()


@app.get("/api/stream/{scan_id}")
async def stream_scan(scan_id: str):
    if scan_id not in scans:
        raise HTTPException(status_code=404, detail="Scan not found")

    async def event_generator():
        scan = scans[scan_id]
        sent = 0
        max_wait = 600  # 10 minute max for long pipelines
        waited = 0

        while waited < max_wait:
            # Send any new events
            while sent < len(scan["events"]):
                evt = scan["events"][sent]
                sent += 1
                yield {
                    "event": evt["event"],
                    "data": json.dumps(evt["data"], default=str),
                }

            # Check if scan is complete
            if scan["status"] in ("done", "error"):
                # Send final result
                if scan["result"]:
                    yield {
                        "event": "result",
                        "data": json.dumps(scan["result"], default=str),
                    }
                break

            # Wait for new events (using Event object for efficiency)
            scan["new_event"].clear()
            await asyncio.sleep(0.5)
            waited += 0.5

            # Send keepalive comment every 15 seconds
            if waited % 15 < 1:
                yield {"comment": "keepalive"}

    return EventSourceResponse(event_generator(), ping=10)


@app.get("/api/history")
async def get_history():
    output_dir = Path("output")
    runs = []
    for f in sorted(output_dir.glob("run_*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            best = data.get("search_results", {}).get("best_match") or {}
            runs.append({
                "filename": f.name,
                "input_image": data.get("input_image", ""),
                "network": data.get("network", ""),
                "success": data.get("pipeline_success", False),
                "match_found": data.get("search_results", {}).get("success", False),
                "platform": best.get("platform", ""),
                "best_url": best.get("url", ""),
                "score": best.get("composite_score", 0),
                "elapsed": data.get("elapsed_seconds", 0),
                "started_at": data.get("started_at", ""),
            })
        except Exception:
            continue
    return runs


@app.get("/api/result/{filename}")
async def get_result(filename: str):
    filepath = Path("output") / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    data = json.loads(filepath.read_text(encoding="utf-8"))
    return data


@app.get("/api/samples")
async def get_samples():
    samples_dir = Path("samples")
    if not samples_dir.exists():
        return []
    samples = []
    for f in samples_dir.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            samples.append({
                "name": f.name,
                "path": f"/samples/{f.name}",
                "size": f.stat().st_size,
            })
    return samples


if __name__ == "__main__":
    import uvicorn
    print("\n  FaceTrace Web UI")
    print("  ════════════════════════════════════")
    print("  Frontend:  http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("  ════════════════════════════════════\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
