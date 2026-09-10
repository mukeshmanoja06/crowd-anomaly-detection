"""
anomaly_detector.py
--------------------
Optical-flow based crowd anomaly detection.

Approach
========
1. YOLO gives us person bounding boxes + simple tracking IDs (see tracker.py).
2. We compute DENSE OPTICAL FLOW (Farneback) between consecutive frames,
   restricted to the regions where people were detected (the "crowd mask").
3. From the flow field we derive, per-frame, a small set of statistics:
      - mean_motion_magnitude   -> how fast the crowd is moving overall
      - motion_std              -> how *varied* the speed is across the crowd
      - direction_entropy       -> how chaotic/disordered the movement directions are
      - per_track_speed         -> speed of each tracked person (from centroid deltas)
4. These statistics are smoothed with a rolling window (to avoid single noisy
   frames triggering false alarms) and compared against adaptive thresholds
   that are learned from the first N "calibration" frames of the video
   (assumed normal crowd behaviour), plus sensible absolute fallbacks.
5. Anomaly classification rules (heuristic, explainable, tunable):
      RUNNING        -> mean motion magnitude jumps well above baseline
                         while direction stays fairly consistent (coherent fast motion)
      PANIC          -> mean motion magnitude AND motion_std both spike
                         (crowd suddenly fast AND moving at very different speeds/dirs)
      FIGHTING       -> localized cluster of high direction-entropy + high
                         magnitude in a SMALL region with multiple overlapping
                         tracks (i.e. chaotic motion concentrated in a tight area)
      UNUSUAL_MOTION -> generic catch-all when direction_entropy or magnitude
                         deviates strongly (z-score) from the learned baseline
                         but doesn't cleanly match the above patterns.

This is a classical-CV + statistics pipeline (no anomaly-labelled training
data required), which is what makes it deployable out-of-the-box on any
camera/video without per-site training.
"""

import collections
import math
import time

import cv2
import numpy as np


class RollingStats:
    """Maintains a rolling window of scalar values and gives mean/std on demand."""

    def __init__(self, window=30):
        self.window = window
        self.values = collections.deque(maxlen=window)

    def push(self, value):
        self.values.append(value)

    def mean(self):
        if not self.values:
            return 0.0
        return float(np.mean(self.values))

    def std(self):
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values))

    def z_score(self, value, min_std_ratio=0.08, min_std_abs=1e-3):
        """
        z = (value - mean) / std, but std is floored to whichever is larger:
        a small absolute epsilon, or a fraction of the rolling mean's magnitude.
        Without this, a baseline that happens to have near-zero variance
        (e.g. a very steady scene) makes the z-score wildly oversensitive to
        ordinary frame-to-frame noise -- a 0.02 wobble around a mean of 0.06
        would otherwise read as a multi-sigma "anomaly".
        """
        sd = self.std()
        floor = max(min_std_abs, abs(self.mean()) * min_std_ratio)
        sd = max(sd, floor)
        return (value - self.mean()) / sd

    def is_full(self):
        return len(self.values) == self.window


class OpticalFlowAnomalyDetector:
    """
    Computes dense optical flow restricted to person regions and derives
    crowd-level motion statistics used to flag anomalies.
    """

    def __init__(
        self,
        calibration_frames=60,
        history_window=45,
        running_z_thresh=2.0,
        panic_z_thresh=2.5,
        fight_entropy_thresh=0.35,
        fight_min_tracks=2,
        min_mag_abs_floor=1.2,
        debounce_frames=2,
    ):
        self.calibration_frames = calibration_frames
        self.frame_count = 0
        self.debounce_frames = debounce_frames
        self._consecutive_hits = {"RUNNING": 0, "PANIC": 0, "FIGHTING": 0, "UNUSUAL_MOTION": 0}

        # rolling baselines (built mostly from calibration period, kept
        # updating slowly afterwards so the system adapts to lighting/scene)
        self.mag_stats = RollingStats(window=history_window)
        self.std_stats = RollingStats(window=history_window)
        self.entropy_stats = RollingStats(window=history_window)
        self.local_entropy_stats = RollingStats(window=history_window)

        self.running_z_thresh = running_z_thresh
        self.panic_z_thresh = panic_z_thresh
        self.fight_entropy_thresh = fight_entropy_thresh
        self.fight_min_tracks = fight_min_tracks
        self.min_mag_abs_floor = min_mag_abs_floor

        self.prev_gray = None
        self.prev_centroids = {}  # track_id -> (cx, cy)

        # Farneback dense optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    # ------------------------------------------------------------------ #
    # Core per-frame update
    # ------------------------------------------------------------------ #
    def process(self, frame_bgr, boxes, track_ids):
        """
        frame_bgr : current BGR frame (np.ndarray)
        boxes     : list of (x1, y1, x2, y2) person boxes for this frame
        track_ids : list of track ids, same length/order as boxes

        Returns a dict describing this frame's analysis:
        {
          'mean_magnitude', 'motion_std', 'direction_entropy',
          'per_track_speed': {id: speed},
          'anomalies': [ {type, score, boxes, region}, ... ]
        }
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, int(320 * gray.shape[0] / gray.shape[1])))
        scale_x = gray.shape[1] / frame_bgr.shape[1]
        scale_y = gray.shape[0] / frame_bgr.shape[0]

        result = {
            "mean_magnitude": 0.0,
            "motion_std": 0.0,
            "direction_entropy": 0.0,
            "per_track_speed": {},
            "anomalies": [],
        }

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self._update_centroids(boxes, track_ids)
            return result

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        self.prev_gray = gray

        fx, fy = flow[..., 0], flow[..., 1]
        magnitude = np.sqrt(fx**2 + fy**2)
        angle = np.arctan2(fy, fx)

        # Build a crowd mask: only consider flow inside person boxes
        mask = np.zeros(gray.shape, dtype=np.uint8)
        scaled_boxes = []
        for (x1, y1, x2, y2) in boxes:
            sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
            sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)
            sx1, sy1 = max(0, sx1), max(0, sy1)
            sx2, sy2 = min(gray.shape[1], sx2), min(gray.shape[0], sy2)
            if sx2 > sx1 and sy2 > sy1:
                mask[sy1:sy2, sx1:sx2] = 1
                scaled_boxes.append((sx1, sy1, sx2, sy2))

        if mask.sum() == 0:
            self._update_centroids(boxes, track_ids)
            return result

        masked_mag = magnitude[mask == 1]
        masked_ang = angle[mask == 1]

        mean_mag = float(np.mean(masked_mag)) if masked_mag.size else 0.0
        std_mag = float(np.std(masked_mag)) if masked_mag.size else 0.0
        direction_entropy = self._angle_entropy(masked_ang)

        result["mean_magnitude"] = mean_mag
        result["motion_std"] = std_mag
        result["direction_entropy"] = direction_entropy

        # Local (per-cluster) entropy: fighting/scuffles are often a small,
        # localized pocket of chaotic motion that gets averaged away in the
        # GLOBAL entropy if the rest of the crowd is calm. We additionally
        # compute entropy inside each dense cluster of boxes and keep the max.
        local_entropy, local_region, num_clusters, num_hot_clusters = self._max_local_entropy(
            angle, mask, scaled_boxes, scale_x, scale_y
        )
        result["local_entropy"] = local_entropy
        result["local_region"] = local_region

        # Per-track speed from centroid displacement (robust complement
        # to dense flow; good for catching individual sudden sprints)
        per_track_speed = self._compute_track_speeds(boxes, track_ids)
        result["per_track_speed"] = per_track_speed

        # ---- update / use baselines ----
        in_calibration = self.frame_count <= self.calibration_frames
        mag_z = self.mag_stats.z_score(mean_mag)
        std_z = self.std_stats.z_score(std_mag)
        ent_z = self.entropy_stats.z_score(direction_entropy)
        local_ent_z = self.local_entropy_stats.z_score(local_entropy)

        self.mag_stats.push(mean_mag)
        self.std_stats.push(std_mag)
        self.entropy_stats.push(direction_entropy)
        self.local_entropy_stats.push(local_entropy)

        if not in_calibration and self.mag_stats.is_full():
            candidates = self._classify(
                mean_mag, std_mag, direction_entropy,
                mag_z, std_z, ent_z,
                per_track_speed, scaled_boxes, track_ids,
                scale_x, scale_y,
                local_entropy, local_ent_z, local_region,
                num_clusters, num_hot_clusters,
            )
            result["anomalies"] = self._debounce(candidates)

        self._update_centroids(boxes, track_ids)
        return result

    # ------------------------------------------------------------------ #
    def _angle_entropy(self, angles, bins=16):
        """Shannon entropy of the motion-direction histogram, normalized to [0,1].
        High entropy = directions are scattered/chaotic (fight-like, panic-like).
        Low entropy  = everyone moving the same way (coherent flow, e.g. walking)."""
        if angles.size == 0:
            return 0.0
        hist, _ = np.histogram(angles, bins=bins, range=(-math.pi, math.pi))
        prob = hist / (hist.sum() + 1e-9)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob))
        max_entropy = math.log(bins)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def _max_local_entropy(self, angle, mask, scaled_boxes, scale_x, scale_y, radius=60):
        """
        Computes direction entropy *within each dense cluster* of person boxes
        (rather than over the whole crowd) and returns:
          - the highest entropy found and its region (for localizing a fight)
          - the total number of clusters and how many of them are "hot"
            (entropy above the fight threshold) -- this ratio is what lets us
            tell a CONTAINED scuffle (1 hot cluster among several calm ones)
            apart from WIDESPREAD panic (most/all clusters hot at once).
        Returns (best_entropy, region_or_None, num_clusters, num_hot_clusters).
        """
        if not scaled_boxes:
            return 0.0, None, 0, 0

        centers = np.array([
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in scaled_boxes
        ])

        best_entropy = 0.0
        best_region = None

        if len(centers) == 1:
            clusters = [[0]]
        else:
            dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            visited = set()
            clusters = []
            for i in range(len(centers)):
                if i in visited:
                    continue
                group = list(np.where(dists[i] < radius)[0])
                visited.update(group)
                clusters.append(group)

        num_clusters = 0
        num_hot = 0

        for group in clusters:
            if len(group) < 1:
                continue
            xs1 = [scaled_boxes[i][0] for i in group]
            ys1 = [scaled_boxes[i][1] for i in group]
            xs2 = [scaled_boxes[i][2] for i in group]
            ys2 = [scaled_boxes[i][3] for i in group]
            x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)

            local_mask = np.zeros_like(mask)
            local_mask[y1:y2, x1:x2] = 1
            combined = (local_mask == 1) & (mask == 1)
            if combined.sum() < 20:  # too small a sample to trust
                continue

            num_clusters += 1
            local_angles = angle[combined]
            ent = self._angle_entropy(local_angles)
            if ent > self.fight_entropy_thresh:
                num_hot += 1
            if ent > best_entropy:
                best_entropy = ent
                best_region = [
                    int(x1 / scale_x), int(y1 / scale_y),
                    int(x2 / scale_x), int(y2 / scale_y),
                ]

        return best_entropy, best_region, num_clusters, num_hot

    def _compute_track_speeds(self, boxes, track_ids):
        speeds = {}
        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if tid in self.prev_centroids:
                pcx, pcy = self.prev_centroids[tid]
                speeds[tid] = math.hypot(cx - pcx, cy - pcy)
            else:
                speeds[tid] = 0.0
        return speeds

    def _update_centroids(self, boxes, track_ids):
        new_centroids = {}
        for box, tid in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            new_centroids[tid] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.prev_centroids = new_centroids

    def _classify(
        self, mean_mag, std_mag, direction_entropy,
        mag_z, std_z, ent_z,
        per_track_speed, scaled_boxes, track_ids,
        scale_x, scale_y,
        local_entropy, local_ent_z, local_region,
        num_clusters, num_hot_clusters,
    ):
        anomalies = []

        fast_tracks = [tid for tid, s in per_track_speed.items() if s > 8.0]
        abs_fast = mean_mag > self.min_mag_abs_floor and mag_z > 1.0
        coherent_motion = direction_entropy < 0.5 and ent_z < 1.0  # everyone moving the same way

        hot_ratio = (num_hot_clusters / num_clusters) if num_clusters > 0 else 0.0
        # Fighting = chaos CONTAINED to a small part of the crowd (one or a
        # couple of hot clusters among several calm ones). If most/all
        # clusters are hot at once, the disorder is crowd-wide -> panic, not
        # a localized altercation, even though both share high local entropy.
        is_contained_chaos = (
            local_entropy > self.fight_entropy_thresh
            and local_ent_z > 1.5
            and len(scaled_boxes) >= self.fight_min_tracks
            and (num_clusters <= 1 or hot_ratio <= 0.6)
        )
        is_widespread_chaos = (
            local_entropy > self.fight_entropy_thresh
            and num_clusters >= 2
            and hot_ratio > 0.6
        )

        # --- FIGHTING: checked FIRST. A localized cluster of boxes showing high,
        #     statistically unusual direction entropy (chaotic motion concentrated
        #     in a tight area) is a more specific signal than the crowd-wide
        #     averages used for RUNNING/PANIC, and should win even if those
        #     global stats are also somewhat elevated by the same scuffle. ---
        if is_contained_chaos:
            anomalies.append({
                "type": "FIGHTING",
                "label": "Possible Fighting / Altercation",
                "score": round(float(min(1.0, local_ent_z / 4.0)), 2),
                "detail": f"local_entropy={local_entropy:.2f}, local_ent_z={local_ent_z:.2f}",
                "region": local_region,
            })

        # --- RUNNING: magnitude spike, but direction stays coherent (low/normal entropy). ---
        elif mag_z > self.running_z_thresh and coherent_motion:
            anomalies.append({
                "type": "RUNNING",
                "label": "Sudden Running Detected",
                "score": round(float(min(1.0, mag_z / 5.0)), 2),
                "detail": f"mag_z={mag_z:.2f}, fast_tracks={len(fast_tracks)}",
                "track_ids": fast_tracks,
            })

        # --- PANIC: speed spike + spread spike together AND directions are not
        #     uniformly coherent (crowd scatters in different directions at speed),
        #     OR chaotic motion is detected widely across most of the crowd at once. ---
        elif (
            (mag_z > self.panic_z_thresh and std_z > self.panic_z_thresh - 0.5)
            or (abs_fast and std_z > self.panic_z_thresh)
            or is_widespread_chaos
        ):
            anomalies.append({
                "type": "PANIC",
                "label": "Crowd Panic",
                "score": round(float(min(1.0, (mag_z + std_z) / 8.0)), 2),
                "detail": f"mag_z={mag_z:.2f}, std_z={std_z:.2f}, hot_ratio={hot_ratio:.2f}",
            })

        # --- UNUSUAL MOTION: generic statistical outlier fallback ---
        elif max(abs(mag_z), abs(std_z), abs(ent_z)) > 3.0:
            anomalies.append({
                "type": "UNUSUAL_MOTION",
                "label": "Unusual Motion Pattern",
                "score": round(float(min(1.0, max(abs(mag_z), abs(std_z), abs(ent_z)) / 5.0)), 2),
                "detail": f"mag_z={mag_z:.2f}, std_z={std_z:.2f}, ent_z={ent_z:.2f}",
            })

        return anomalies

    def _debounce(self, candidates):
        """
        Requires an anomaly type to appear for `debounce_frames` consecutive
        frames before it's actually reported. This suppresses one-off noise
        spikes (a single jittery frame) while still reacting within a fraction
        of a second of real, sustained anomalous motion.
        """
        candidate_types = {a["type"] for a in candidates}
        confirmed = []

        for atype in self._consecutive_hits:
            if atype in candidate_types:
                self._consecutive_hits[atype] += 1
            else:
                self._consecutive_hits[atype] = 0

        for a in candidates:
            if self._consecutive_hits[a["type"]] >= self.debounce_frames:
                confirmed.append(a)

        return confirmed
