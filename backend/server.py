"""
server.py
---------
FastAPI backend that:
  1. Serves the HTML/CSS/JS frontend (static files).
  2. Accepts an uploaded video file OR a webcam/RTSP/CCTV source string.
  3. Streams processed (annotated) frames as base64 JPEG over a WebSocket,
     along with JSON metadata (people count, anomaly alerts, stats).
  4. Exposes a REST endpoint to fetch the rolling alert log.

Run with:
    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser.
"""

import asyncio
import base64
import json
import os
import shutil
import time
import uuid

import cv2
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .pipeline import CrowdAnomalyPipeline
except ImportError:
    from pipeline import CrowdAnomalyPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "..", "sample_data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Crowd Anomaly Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Holds active pipeline + capture per session id so multiple browser tabs
# don't fight over the same state.
SESSIONS = {}


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/config.js")
async def frontend_config():
    return FileResponse(os.path.join(FRONTEND_DIR, "config.js"), media_type="application/javascript")


@app.get("/app.js")
async def frontend_app():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"), media_type="application/javascript")


@app.get("/style.css")
async def frontend_styles():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"), media_type="text/css")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save an uploaded video file and return a session_id + source path
    that the websocket endpoint can be told to open."""
    ext = os.path.splitext(file.filename)[1] or ".mp4"
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"source": fpath, "filename": file.filename})


@app.get("/api/sources")
async def list_sources():
    """Helpful info for the frontend: what source types are supported."""
    return {
        "webcam": "0 (or any integer index of a local webcam)",
        "rtsp_example": "rtsp://username:password@camera-ip:554/stream",
        "http_example": "http://camera-ip:8080/video",
        "note": "For a real CCTV feed, paste its RTSP/HTTP URL into the 'Live Feed URL' field.",
    }


@app.websocket("/ws/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    pipeline = None
    cap = None
    session_id = uuid.uuid4().hex

    try:
        # First message from client must specify the source:
        # {"source": "<path or 0 or rtsp url>", "model": "yolo11n.pt", "conf": 0.35}
        init_msg = await websocket.receive_text()
        cfg = json.loads(init_msg)
        source = cfg.get("source", 0)
        model_path = cfg.get("model", "yolo11n.pt")
        conf = float(cfg.get("conf", 0.35))

        # numeric strings -> int (webcam index)
        if isinstance(source, str) and source.strip().lstrip("-").isdigit():
            source = int(source)

        pipeline = CrowdAnomalyPipeline(model_path=model_path, conf=conf)
        SESSIONS[session_id] = {"pipeline": pipeline, "running": True}

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Could not open video source: {source}",
            }))
            return

        await websocket.send_text(json.dumps({"type": "started", "session_id": session_id}))

        frame_skip = cfg.get("frame_skip", 0)  # process every Nth frame for speed
        count = 0

        while True:
            # allow client to send control messages (e.g. stop) without blocking
            try:
                control = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(control)
                if msg.get("action") == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break

            ret, frame = cap.read()
            if not ret:
                await websocket.send_text(json.dumps({"type": "ended"}))
                break

            count += 1
            if frame_skip and count % (frame_skip + 1) != 0:
                continue

            annotated, info = pipeline.process_frame(frame)

            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                continue
            b64 = base64.b64encode(buf).decode("utf-8")

            payload = {
                "type": "frame",
                "image": b64,
                "info": info,
            }
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0)  # yield control

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        if cap is not None:
            cap.release()
        SESSIONS.pop(session_id, None)
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
