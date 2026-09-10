"""
run_cli.py
----------
Standalone command-line tool: processes a video file end-to-end (YOLO11
person detection + optical-flow anomaly detection) and writes an annotated
output video plus a JSON alert log. Useful for quick testing without
starting the web server, or for batch-processing recorded CCTV footage.

Usage:
    python run_cli.py --input path/to/video.mp4 --output outputs/result.mp4
    python run_cli.py --input 0                     # webcam, press 'q' to quit, preview window
    python run_cli.py --input rtsp://... --output outputs/cctv_result.mp4

Args:
    --input         Video file path, webcam index (e.g. 0), or RTSP/HTTP URL.
    --output        Path to save annotated output video (.mp4). Optional.
    --model         YOLO weights file (default: yolo11n.pt, auto-downloaded).
    --conf          Detection confidence threshold (default: 0.35).
    --no-preview    Disable the live cv2 preview window (useful on servers).
    --calib-frames  Number of initial frames used to learn the 'normal' baseline.
"""

import argparse
import json
import os
import sys

import cv2

try:
    from .pipeline import CrowdAnomalyPipeline
except ImportError:
    from pipeline import CrowdAnomalyPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Crowd anomaly detection (batch/CLI mode)")
    p.add_argument("--input", required=True, help="video file path, webcam index, or stream URL")
    p.add_argument("--output", default=None, help="path to save annotated output video")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights file")
    p.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    p.add_argument("--no-preview", action="store_true", help="disable live preview window")
    p.add_argument("--calib-frames", type=int, default=60, help="baseline calibration frame count")
    return p.parse_args()


def main():
    args = parse_args()

    source = args.input
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    pipeline = CrowdAnomalyPipeline(
        model_path=args.model, conf=args.conf, calibration_frames=args.calib_frames
    )

    print(f"[INFO] Processing source: {args.input}  (model={args.model}, conf={args.conf})")
    print(f"[INFO] Calibrating baseline for first {args.calib_frames} frames -- "
          f"keep this period 'normal' crowd behaviour for best results.")

    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            annotated, info = pipeline.process_frame(frame)

            if info["anomalies"]:
                for a in info["anomalies"]:
                    print(f"[ALERT] frame={frame_no} type={a['type']} "
                          f"label='{a['label']}' score={a['score']}")

            if writer is not None:
                writer.write(annotated)

            if not args.no_preview:
                cv2.imshow("Crowd Anomaly Detection (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        log_path = None
        if args.output:
            log_path = os.path.splitext(args.output)[0] + "_alerts.json"
            with open(log_path, "w") as f:
                json.dump(pipeline.alert_log, f, indent=2)

        print(f"[DONE] Processed {frame_no} frames.")
        if args.output:
            print(f"[DONE] Annotated video saved to: {args.output}")
            print(f"[DONE] Alert log saved to: {log_path}")


if __name__ == "__main__":
    main()
