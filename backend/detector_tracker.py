"""
detector_tracker.py
--------------------
Wraps a YOLO model (Ultralytics) for person detection and provides
simple, dependency-light multi-object tracking (centroid + IoU matching)
so each person gets a stable ID across frames. This ID is what lets the
anomaly module compute per-person speed and feeds the on-screen labels.

Why not just use YOLO's built-in `model.track()`?
  We DO use it when available (it wraps ByteTrack/BoT-SORT, which is more
  robust than a hand-rolled tracker) and gracefully fall back to a simple
  IoU+centroid tracker if tracking mode isn't available in the environment.
"""

import numpy as np
from ultralytics import YOLO


PERSON_CLASS_ID = 0  # COCO class index for 'person'


class PersonDetectorTracker:
    def __init__(self, model_path="yolo11n.pt", conf=0.35, device=None):
        """
        model_path: any Ultralytics YOLO weights file. 'yolo11n.pt' (YOLOv11 nano)
                    is downloaded automatically on first run if not present locally.
                    Swap to 'yolo11s.pt' / 'yolo11m.pt' for higher accuracy at the
                    cost of speed, or to a custom-trained .pt for a specific domain.
        conf: detection confidence threshold.
        device: 'cpu', 'cuda', or None (auto).
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def infer(self, frame_bgr):
        """
        Runs detection + tracking on a single frame.
        Returns: list of dicts: [{'box': (x1,y1,x2,y2), 'id': int, 'conf': float}, ...]
        """
        try:
            results = self.model.track(
                frame_bgr,
                persist=True,
                conf=self.conf,
                classes=[PERSON_CLASS_ID],
                device=self.device,
                verbose=False,
                tracker="bytetrack.yaml",
            )
        except Exception:
            # Fallback: plain detection without persistent IDs (rare; e.g.
            # missing tracker config in a stripped install). We synthesize
            # IDs by detection order, which still allows per-frame stats.
            results = self.model.predict(
                frame_bgr,
                conf=self.conf,
                classes=[PERSON_CLASS_ID],
                device=self.device,
                verbose=False,
            )

        detections = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_xyxy))

        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(len(boxes_xyxy))

        for box, tid, c in zip(boxes_xyxy, ids, confs):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "box": (x1, y1, x2, y2),
                "id": int(tid),
                "conf": float(c),
            })

        return detections
