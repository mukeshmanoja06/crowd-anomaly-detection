# SENTRY — Vision-Based Crowd Anomaly Detection

A working, end-to-end system that watches a video feed (recorded file or live
camera/CCTV stream), detects people with **YOLOv11**, tracks them across
frames, and flags **crowd anomalies** — sudden running, panic, fighting,
and other unusual motion — using a **dense optical-flow** analysis engine.
A dark, console-style web dashboard (HTML/CSS/JS) shows the live annotated
video plus real-time telemetry and an alert log.

```
crowd-anomaly-detection/
├── backend/
│   ├── detector_tracker.py   # YOLOv11 person detection + tracking
│   ├── anomaly_detector.py   # optical-flow anomaly engine (the core logic)
│   ├── pipeline.py           # glues detection + anomaly + drawing together
│   ├── server.py             # FastAPI + WebSocket server (powers the web UI)
│   └── run_cli.py            # standalone CLI — no web server needed
├── frontend/
│   ├── index.html            # dashboard layout
│   ├── style.css             # dark "surveillance console" theme
│   └── app.js                # WebSocket client, live telemetry rendering
├── requirements.txt
└── README.md
```

---

## 1. How it works

**Detection & tracking** (`detector_tracker.py`)
Every frame is passed to a YOLOv11 model (`ultralytics` package) filtered to
the `person` class. Ultralytics' built-in `model.track()` (ByteTrack) assigns
a persistent ID to each person so we can follow them across frames.

**Anomaly detection** (`anomaly_detector.py`) — the chosen method is
**dense optical flow (Farneback)**, restricted to the pixels inside detected
person boxes ("crowd mask"), reduced to a few interpretable statistics each
frame:

| Statistic | What it captures |
|---|---|
| `mean_magnitude` | how fast the crowd is moving overall |
| `motion_std` | how *varied* people's speeds are (panic = high variance) |
| `direction_entropy` | how chaotic the movement directions are (0 = everyone moving the same way, 1 = totally random) |
| `local_entropy` | direction entropy computed **inside each spatial cluster** of people, so a small scuffle isn't averaged away by a calm crowd around it |

The first `calibration_frames` (default 60) frames are used to learn a
**per-video baseline** (rolling mean/std) of "normal" crowd behaviour — no
labeled training data needed. After that, each new frame's statistics are
converted to z-scores against that baseline and run through a small,
explainable rule set:

- **RUNNING** — speed z-score spikes, but everyone's still moving in a
  coherent direction (an orderly sprint, not a scatter).
- **FIGHTING** — one tight spatial cluster shows high, statistically unusual
  direction entropy while the rest of the crowd is calm (localized scuffle).
- **PANIC** — speed AND variance both spike together, or chaotic motion is
  detected across *most* of the crowd at once (widespread, not contained).
- **UNUSUAL_MOTION** — generic catch-all for any other strong statistical
  outlier that doesn't cleanly match the patterns above.

Each anomaly type requires **2 consecutive frames** of the same signal before
it's reported (debouncing), so single noisy frames don't trigger false alarms.

**Why optical flow instead of a trained autoencoder/LSTM?** It needs zero
labeled "anomaly" training data (which is scarce/sensitive for fighting and
panic footage), runs in real time on CPU, and self-calibrates to each new
camera/scene during the first ~60 frames.

**Pipeline & drawing** (`pipeline.py`) ties detection + anomaly detection
together and draws bounding boxes (green = normal, red = flagged), highlights
the anomaly region, and shows a status bar + alert banner on the frame.

**Server** (`server.py`) is a FastAPI app that serves the frontend and
streams annotated frames + JSON telemetry over a WebSocket
(`/ws/stream`) as base64 JPEGs, so the browser shows a live "feed".

**CLI** (`run_cli.py`) runs the exact same pipeline without any web server —
useful for quickly testing on a video file and saving an annotated output.

---

## 2. Setup

### Requirements
- Python 3.9–3.12
- pip
- (Optional) an NVIDIA GPU + CUDA for faster inference — CPU works fine for
  testing, just slower.

### Install

```bash
cd crowd-anomaly-detection
pip install -r requirements.txt
```

This installs: `ultralytics` (YOLOv11), `opencv-python`, `numpy`, `fastapi`,
`uvicorn`, `python-multipart`.

> **First run will auto-download the YOLO weights.** The first time you run
> the detector, `ultralytics` automatically downloads `yolo11n.pt` (~5–6 MB)
> from GitHub. This requires normal internet access — if you're behind a
> restrictive firewall/proxy that blocks `github.com` /
> `release-assets.githubusercontent.com`, the download will fail; either
> allow that domain or manually download the weights from
> https://github.com/ultralytics/assets/releases and place the `.pt` file in
> the `backend/` folder (or pass its path via `--model`).

---

## 3. Running it

### Option A — Web dashboard (recommended)

From the project root, run:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000
```

The equivalent backend-directory command is:

```bash
cd backend
uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open **http://localhost:8000** in your browser. In the dashboard:

1. **Upload a video file** (drag-and-drop or click the upload box), **or**
   type a **live source** into "Live feed URL":
   - `0` → your computer's default webcam
   - `rtsp://user:pass@camera-ip:554/stream` → an RTSP CCTV camera
   - `http://camera-ip:8080/video` → an HTTP/MJPEG camera stream
2. Pick a model size (Nano = fastest, Medium = most accurate) and confidence
   threshold.
3. Click **Start monitoring**. The video panel shows the live annotated feed;
   the right panel shows live motion telemetry and the alert log.
4. The first ~60 frames calibrate the "normal" baseline for that specific
   video/camera — keep that window free of unusual activity for the best
   results, just like a real surveillance system warming up.

### Option B — Command line (no browser needed)

Process a video file and save an annotated copy:

```bash
cd backend
python run_cli.py --input /path/to/video.mp4 --output ../outputs/result.mp4
```

Live webcam, with a preview window (press `q` to quit):

```bash
python run_cli.py --input 0
```

RTSP/CCTV stream:

```bash
python run_cli.py --input "rtsp://user:pass@192.168.1.50:554/stream" --output ../outputs/cctv_result.mp4
```

Useful flags:
```
--model yolo11s.pt     # use a bigger/more accurate model (n/s/m/l/x)
--conf 0.4             # detection confidence threshold
--calib-frames 90      # longer calibration window for noisier scenes
--no-preview           # disable the cv2 window (e.g. running on a server)
```

Output: an annotated `.mp4` plus a `<output>_alerts.json` file logging every
anomaly detected (frame number, type, score, timestamp).

---

## 4. Testing without real CCTV footage

Any video with people walking works for a basic test. To see the anomaly
types fire, try clips that include:
- a calm period first (so the baseline calibrates on "normal" behaviour)
- people suddenly running
- a crowd scattering quickly
- a scuffle/fight in a small area

Stock footage sites (Pexels, Pixabay) have free "crowd walking" and
"people running" clips suitable for a quick demo.

---

## 5. Deploying for access from any device

This project uses two services in production:

- **Netlify** hosts the static dashboard in `frontend/`.
- **Render** runs the Python FastAPI backend, YOLO detector, OpenCV pipeline,
  uploads, and WebSocket stream.

### Deploy the backend on Render

1. Push this repository to GitHub.
2. In Render, choose **New > Blueprint** and select the repository.
3. Render detects `render.yaml` and creates the `sentry-crowd-anomaly-api`
  service.
4. Copy the deployed backend URL, such as
  `https://sentry-crowd-anomaly-api.onrender.com`.

The first request downloads `yolo11n.pt`, so the first startup can take longer.
Use a paid Render instance for reliable always-on processing; free instances
may sleep when idle.

### Deploy the dashboard on Netlify

1. In Netlify, choose **Add new site > Import an existing project**.
2. Select the same GitHub repository.
3. Netlify detects `netlify.toml`; the publish directory is `frontend`.
4. Before deploying, edit `frontend/config.js` and set
  `window.SENTRY_API_URL` to the public Render backend URL.
5. Deploy the site and open its `https://...netlify.app` URL from any device.

The frontend automatically uses `wss://` for the WebSocket when served over
HTTPS. Keep the Render backend URL on HTTPS as well.

---

## 6. Tuning

All thresholds live at the top of `OpticalFlowAnomalyDetector.__init__` in
`anomaly_detector.py`:

| Parameter | Effect |
|---|---|
| `calibration_frames` | how many frames to learn "normal" before alerting |
| `running_z_thresh` | lower = more sensitive to speed spikes |
| `panic_z_thresh` | lower = more sensitive to panic-like dispersal |
| `fight_entropy_thresh` | lower = more sensitive to localized chaotic motion |
| `debounce_frames` | how many consecutive frames are needed before alerting |

If you get false alarms on a specific camera, try raising the relevant
threshold or extending `calibration_frames`; if real events are missed,
lower the thresholds.

---

## 7. Extending this project

- Swap `yolo11n.pt` for `yolo11s.pt`/`yolo11m.pt` for better accuracy.
- Add an LSTM/autoencoder on top of the existing optical-flow feature stream
  (`mean_magnitude`, `motion_std`, `direction_entropy` per frame) if you
  later have labeled anomaly data — the features are already extracted, so
  this would be a drop-in replacement for the rule-based `_classify` step.
- Add e-mail/SMS/webhook alerting from `pipeline.py`'s `alert_log`.
- Persist alerts to a database instead of the in-memory rolling log.
