"""
pipeline.py
-----------
Glues together: PersonDetectorTracker (YOLO11) -> OpticalFlowAnomalyDetector
-> frame annotation (boxes, IDs, alert banners) -> JPEG-encoded output.

This module is UI-agnostic: it's used by both the FastAPI server (for live
streaming to the browser) and could equally be used from a plain CLI script
(see run_cli.py) for batch processing a video file to an output .mp4.
"""

import time
import cv2
import numpy as np

try:
    from .detector_tracker import PersonDetectorTracker
    from .anomaly_detector import OpticalFlowAnomalyDetector
except ImportError:
    from detector_tracker import PersonDetectorTracker
    from anomaly_detector import OpticalFlowAnomalyDetector


ANOMALY_COLORS = {
    "RUNNING": (0, 165, 255),       # orange
    "PANIC": (0, 0, 255),           # red
    "FIGHTING": (0, 0, 139),        # dark red
    "UNUSUAL_MOTION": (0, 255, 255),  # yellow
}


class CrowdAnomalyPipeline:
    def __init__(self, model_path="yolo11n.pt", conf=0.35, device=None,
                 calibration_frames=60):
        self.detector = PersonDetectorTracker(model_path=model_path, conf=conf, device=device)
        self.anomaly_detector = OpticalFlowAnomalyDetector(calibration_frames=calibration_frames)
        self.frame_idx = 0
        self.alert_log = []  # rolling log of {frame, time, type, label, score}
        self.fps_estimate = 0.0
        self._last_t = time.time()

    def process_frame(self, frame_bgr):
        """
        Runs the full pipeline on one frame.
        Returns: (annotated_frame_bgr, info_dict)
        info_dict contains people_count, anomalies, calibration status, fps.
        """
        now = time.time()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            self.fps_estimate = 0.9 * self.fps_estimate + 0.1 * (1.0 / dt)

        self.frame_idx += 1
        detections = self.detector.infer(frame_bgr)
        boxes = [d["box"] for d in detections]
        ids = [d["id"] for d in detections]

        analysis = self.anomaly_detector.process(frame_bgr, boxes, ids)

        annotated = self._draw(frame_bgr, detections, analysis)

        in_calibration = self.anomaly_detector.frame_count <= self.anomaly_detector.calibration_frames

        for a in analysis["anomalies"]:
            self.alert_log.append({
                "frame": self.frame_idx,
                "time": round(now, 2),
                "type": a["type"],
                "label": a["label"],
                "score": a["score"],
            })
        # keep log bounded
        if len(self.alert_log) > 200:
            self.alert_log = self.alert_log[-200:]

        info = {
            "frame_idx": self.frame_idx,
            "people_count": len(detections),
            "anomalies": analysis["anomalies"],
            "mean_magnitude": round(analysis["mean_magnitude"], 3),
            "motion_std": round(analysis["motion_std"], 3),
            "direction_entropy": round(analysis["direction_entropy"], 3),
            "calibrating": in_calibration,
            "fps": round(self.fps_estimate, 1),
        }
        return annotated, info

    def _draw(self, frame, detections, analysis):
        out = frame.copy()
        h, w = out.shape[:2]

        anomaly_types_present = {a["type"] for a in analysis["anomalies"]}
        fast_ids = set()
        for a in analysis["anomalies"]:
            if "track_ids" in a:
                fast_ids.update(a["track_ids"])

        # person boxes
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            tid = d["id"]
            is_flagged = tid in fast_ids
            color = (0, 0, 255) if is_flagged else (0, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"ID {tid}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # anomaly region highlight (e.g. fight cluster)
        for a in analysis["anomalies"]:
            region = a.get("region")
            if region:
                rx1, ry1, rx2, ry2 = region
                color = ANOMALY_COLORS.get(a["type"], (255, 255, 255))
                cv2.rectangle(out, (rx1, ry1), (rx2, ry2), color, 3)
                cv2.putText(out, a["label"], (rx1, max(0, ry1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # top status bar
        cv2.rectangle(out, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.putText(out, f"People: {len(detections)}", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if self.anomaly_detector.frame_count <= self.anomaly_detector.calibration_frames:
            cv2.putText(out, "CALIBRATING BASELINE...", (160, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # alert banner
        if analysis["anomalies"]:
            top = max(analysis["anomalies"], key=lambda a: a["score"])
            color = ANOMALY_COLORS.get(top["type"], (0, 0, 255))
            cv2.rectangle(out, (0, h - 40), (w, h), color, -1)
            cv2.putText(
                out,
                f"ALERT: {top['label']}  (score {top['score']})",
                (10, h - 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return out
