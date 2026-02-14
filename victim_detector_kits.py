#python victim_detector_kits.py
# victim_detector_kits.py
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict

import cv2
from ultralytics import YOLO

# ===================== SETTINGS =====================
MODEL_PATH = "best.pt"
CAM_INDEX = 0
IMGSZ = 640

# --- Counting gate: must be visible this long before it can count ---
REQUIRED_SEE_SEC = 2.0

# Allowed tiny flicker while still “continuous”
COUNT_DROPOUT_GRACE_SEC = 0.35

# Main tracking threshold (keep candidate detections)
TRACK_CONF = 0.75

# Model predict conf (low so we can still see alt labels for voting)
MODEL_PREDICT_CONF = 0.10

# Class-specific scoring confidence (Ω higher to reduce Φ->Ω mistakes)
CLASS_SCORE_CONF = {"phi": 0.75, "psi": 0.75, "omega": 0.85}

# Label must dominate the recent window to be accepted
CLASS_SCORE_CONF["omega"] = 0.87
LABEL_DOMINANCE_MIN = 0.80

# If the sign disappears briefly, we should NOT recount
SHORT_DROPOUT_GRACE_SEC = 8.0

# If the sign is gone longer than this, it can count as NEW again
RECOUNT_AFTER_ABSENCE_SEC = 20.0   # set huge for “never recount”

# IMPORTANT: only refresh memory when we have strong/stable detections
MEMORY_TOUCH_CONF = 0.75
MEMORY_TOUCH_MIN_STABLE_SEC = 0.50

# Filters to kill false positives
MIN_BOX_AREA_FRAC = 0.012  # 1.2% of frame area

# Matching thresholds
MATCH_IOU_MIN = 0.20
MATCH_DIST_FRAC = 0.22     # 22% of min(frame_w, frame_h)

# If two boxes overlap (possibly different labels), vote them together
CLUSTER_IOU = 0.35

# ===================== KIT RULES (EDIT IF NEEDED) =====================
# Keeping YOUR mapping — adjust if your final rules define letters differently.
KITS_BY_CLASS = {"phi": 2, "psi": 1, "omega": 0}
POINTS_BY_KITS = {0: 0, 1: 10, 2: 30}
KIT_BUDGET = 8
# ==========================================================


def normalize_label(s: str) -> str:
    s = (s or "").strip()
    low = s.lower()
    # Handle english names + greek characters
    if low in {"phi", "φ", "Φ".lower()} or s in {"Φ", "φ"}:
        return "phi"
    if low in {"psi", "ψ", "Ψ".lower()} or s in {"Ψ", "ψ"}:
        return "psi"
    if low in {"omega", "ω", "Ω".lower()} or s in {"Ω", "ω"}:
        return "omega"
    return low


def box_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = box_area(a)
    area_b = box_area(b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def match_bbox(a_bbox, b_bbox, W, H) -> bool:
    i = iou_xyxy(a_bbox, b_bbox)
    d = dist(center(a_bbox), center(b_bbox))
    max_d = MATCH_DIST_FRAC * min(W, H)
    return (i >= MATCH_IOU_MIN) or (d <= max_d)


def smooth_bbox(oldb, newb, alpha=0.75):
    return tuple(alpha * o + (1 - alpha) * n for o, n in zip(oldb, newb))


def cluster_detections(dets):
    """
    dets: list of (label, conf, bbox)
    Returns clusters: list of list indices
    """
    clusters = []
    used = [False] * len(dets)

    for i in range(len(dets)):
        if used[i]:
            continue
        used[i] = True
        cluster = [i]
        changed = True
        while changed:
            changed = False
            for j in range(len(dets)):
                if used[j]:
                    continue
                # if overlaps any in cluster -> join
                for k in cluster:
                    if iou_xyxy(dets[j][2], dets[k][2]) >= CLUSTER_IOU:
                        used[j] = True
                        cluster.append(j)
                        changed = True
                        break
        clusters.append(cluster)
    return clusters


def pick_best_cluster(dets):
    """
    Groups overlapping boxes (even with different labels) and votes labels by summed confidence.
    Returns:
      (voted_label, voted_support, best_conf, best_bbox, label_sums)
    """
    if not dets:
        return None

    clusters = cluster_detections(dets)
    best = None

    for cl in clusters:
        label_sums = defaultdict(float)
        best_conf = -1.0
        best_bbox = None

        for idx in cl:
            label, conf, bbox = dets[idx]
            label_sums[label] += conf
            if conf > best_conf:
                best_conf = conf
                best_bbox = bbox

        total = sum(label_sums.values()) + 1e-9
        voted_label = max(label_sums.items(), key=lambda x: x[1])[0]
        voted_support = label_sums[voted_label] / total  # 0..1
        cluster_score = total  # overall strength

        if (best is None) or (cluster_score > best[0]):
            best = (cluster_score, voted_label, voted_support, best_conf, best_bbox, dict(label_sums))

    _, voted_label, voted_support, best_conf, best_bbox, label_sums = best
    return voted_label, voted_support, best_conf, best_bbox, label_sums


@dataclass
class MemoryVictim:
    bbox: Tuple[float, float, float, float]
    last_seen_strong: float  # updated only by strong/stable detections


@dataclass
class Track:
    bbox: Tuple[float, float, float, float]
    last_seen: float
    continuous_start: float
    counted: bool = False

    # store recent observations: (t, label, conf, support)
    obs: deque = field(default_factory=lambda: deque(maxlen=200))

    def reset_continuous(self, now: float):
        self.continuous_start = now
        self.obs.clear()
        self.counted = False


def compute_stable_label(track: Track, now: float):
    """
    Uses observations since continuous_start (or last REQUIRED_SEE_SEC) to decide label.
    Returns: (stable_label, dominance, avg_conf)
    """
    # Only use obs within the last REQUIRED_SEE_SEC window
    window_start = max(track.continuous_start, now - REQUIRED_SEE_SEC)
    sums = defaultdict(float)
    counts = defaultdict(int)

    for (t, label, conf, _support) in track.obs:
        if t >= window_start:
            sums[label] += conf
            counts[label] += 1

    total = sum(sums.values()) + 1e-9
    if total <= 1e-8:
        return None, 0.0, 0.0

    stable_label = max(sums.items(), key=lambda x: x[1])[0]
    dominance = sums[stable_label] / total
    avg_conf = sums[stable_label] / max(1, counts[stable_label])
    return stable_label, dominance, avg_conf


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    memory: List[MemoryVictim] = []
    track: Optional[Track] = None

    counts = {k: 0 for k in KITS_BY_CLASS}
    total_kits = 0
    total_points = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        frame_area = float(H * W)
        now = time.time()

        # Expire memory after long absence
        memory = [m for m in memory if (now - m.last_seen_strong) <= RECOUNT_AFTER_ABSENCE_SEC]

        # Predict
        res = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=MODEL_PREDICT_CONF,
            verbose=False
        )[0]

        # Build detections list (keep moderately confident boxes for voting)
        dets = []
        if res.boxes is not None:
            for b in res.boxes:
                conf = float(b.conf[0])
                if conf < 0.25:
                    continue

                cls_id = int(b.cls[0])
                raw = res.names[cls_id]
                label = normalize_label(raw)

                if label not in KITS_BY_CLASS:
                    continue
                if conf < TRACK_CONF:
                    continue

                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)

                if box_area(bbox) < MIN_BOX_AREA_FRAC * frame_area:
                    continue

                dets.append((label, conf, bbox))

        best = pick_best_cluster(dets)  # voted_label, voted_support, best_conf, best_bbox, label_sums

        # Update/maintain track
        if best is None:
            if track is not None and (now - track.last_seen) <= SHORT_DROPOUT_GRACE_SEC:
                # keep track alive
                pass
            else:
                track = None
        else:
            voted_label, voted_support, best_conf, best_bbox, _label_sums = best

            if track is None:
                track = Track(bbox=best_bbox, last_seen=now, continuous_start=now)
            else:
                # Same physical object? (label-agnostic bbox match)
                if match_bbox(track.bbox, best_bbox, W, H):
                    # continuous?
                    if (now - track.last_seen) > COUNT_DROPOUT_GRACE_SEC:
                        track.reset_continuous(now)
                    track.bbox = smooth_bbox(track.bbox, best_bbox)
                    track.last_seen = now
                else:
                    # New object candidate
                    track = Track(bbox=best_bbox, last_seen=now, continuous_start=now)

            # store observation for voting
            track.obs.append((now, voted_label, best_conf, voted_support))

            # Draw bbox + live label
            x1, y1, x2, y2 = map(int, best_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"live={voted_label} conf={best_conf:.2f} vote={voted_support:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            # Touch memory only when detection is stable & strong (prevents ghosts refreshing)
            stable_label, dom, avgc = compute_stable_label(track, now)
            stable_sec = now - track.continuous_start
            if stable_label is not None and stable_sec >= MEMORY_TOUCH_MIN_STABLE_SEC and best_conf >= MEMORY_TOUCH_CONF:
                for m in memory:
                    if match_bbox(m.bbox, track.bbox, W, H):
                        m.last_seen_strong = now
                        m.bbox = smooth_bbox(m.bbox, track.bbox)
                        break

        # ===== Counting logic (2 seconds + dominance + class-specific confidence) =====
        if track is not None and not track.counted:
            stable_label, dom, avgc = compute_stable_label(track, now)
            stable_sec = now - track.continuous_start

            if stable_label is not None:
                needed_conf = CLASS_SCORE_CONF.get(stable_label, 0.75)

                eligible = (
                    stable_sec >= REQUIRED_SEE_SEC and
                    dom >= LABEL_DOMINANCE_MIN and
                    avgc >= needed_conf
                )

                if eligible:
                    # Already counted? (label-agnostic memory)
                    already = False
                    for m in memory:
                        if match_bbox(m.bbox, track.bbox, W, H):
                            already = True
                            break

                    if not already:
                        kits = KITS_BY_CLASS[stable_label]
                        pts = POINTS_BY_KITS.get(kits, 0)

                        counts[stable_label] += 1
                        total_kits += kits
                        total_points += pts

                        memory.append(MemoryVictim(track.bbox, now))
                        track.counted = True

                        print(
                            f"[COUNT NEW] {stable_label.upper()} -> DROP {kits} kits | "
                            f"seen={stable_sec:.2f}s dom={dom:.2f} avgc={avgc:.2f} | "
                            f"total_kits={total_kits} | points={total_points}"
                        )

        # ===== HUD =====
        y = 30
        cv2.putText(frame, f"TOTAL KITS: {total_kits} (budget {KIT_BUDGET})", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 32
        cv2.putText(frame, f"KIT POINTS: {total_points}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 36
        cv2.putText(frame, f"phi:{counts['phi']}  psi:{counts['psi']}  omega:{counts['omega']}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        y += 32

        if track is None:
            cv2.putText(frame, "TRACK: none", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            stable_label, dom, avgc = compute_stable_label(track, now)
            seen = now - track.continuous_start
            if stable_label is None:
                txt = f"TRACK: seen={seen:.2f}s (gathering...)"
            else:
                need = CLASS_SCORE_CONF.get(stable_label, 0.75)
                txt = (f"TRACK: stable={stable_label} seen={seen:.2f}/{REQUIRED_SEE_SEC}s "
                       f"dom={dom:.2f}/{LABEL_DOMINANCE_MIN} avgc={avgc:.2f}/{need}")
            cv2.putText(frame, txt, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.putText(frame, f"RecountAfter={RECOUNT_AFTER_ABSENCE_SEC}s  TrackConf={TRACK_CONF}",
                    (20, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("RCJ 2026 Kits - 2s Stable Count", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            memory.clear()
            track = None
            counts = {k: 0 for k in KITS_BY_CLASS}
            total_kits = 0
            total_points = 0
            print("[RESET] cleared counts/memory")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
