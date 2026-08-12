"""Oil-drop detection and tracking: video/frame-dir → per-frame drop coordinate CSV.

Pipeline: median background modeling → background subtraction → threshold/morphology
→ blob detection → gated tracking.

Usage examples:
    python detect.py --input frames_sample18/ --output drop_pixel_coords.csv
    python detect.py --input movie.mp4 --output drop_pixel_coords.csv --fps 60
"""
import argparse
import csv
import math
import os

import cv2

from config import Config, apply_overrides, build_common_parser
from utils import create_median_background, iter_frames


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[build_common_parser()],
        description="Oil-drop detection and tracking: outputs per-frame drop coordinate CSV",
    )
    parser.add_argument("--bg-samples", type=int, default=None, dest="num_bg_samples",
                        help="Number of frames sampled for background modeling")
    parser.add_argument("--threshold", type=int, default=None, dest="threshold_val",
                        help="Binary subtraction threshold; lower is more sensitive")
    parser.add_argument("--min-area", type=int, default=None, dest="min_area")
    parser.add_argument("--max-area", type=int, default=None, dest="max_area")
    parser.add_argument("--circularity", type=float, default=None, dest="circularity_min",
                        help="Minimum blob circularity")
    parser.add_argument("--gate-radius", type=float, default=None, dest="gate_radius_min_px",
                        help="Minimum search-gate radius (pixels)")
    parser.add_argument("--max-missed", type=int, default=None, dest="max_missed_frames",
                        help="Consecutive missed frames before tracking resets")
    parser.add_argument("--ema-alpha", type=float, default=None, dest="ema_alpha",
                        help="Velocity smoothing factor (0,1]; larger follows measurement more")
    parser.add_argument("--reinit-rule", choices=("largest_blob", "highest_conf", "nearest_center"),
                        default=None, dest="reinit_rule",
                        help="Strategy for choosing the target blob on (re)initialization")
    parser.add_argument("--no-bg-debug", action="store_true", help="Do not save the background debug image")
    return parser


def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter * perimeter)


def detect_candidates(bw, min_area, max_area, circularity_min, img_w, img_h):
    """Extract blob candidates from a binary image, scoring confidence and center distance."""
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_x, center_y = img_w / 2.0, img_h / 2.0
    sigma2 = (min(img_w, img_h) / 3.0) ** 2

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        circ = compute_circularity(cnt)
        if circ < circularity_min:
            continue
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        confidence = circ * (area / max_area)
        dist2 = (cx - center_x) ** 2 + (cy - center_y) ** 2
        score = confidence * math.exp(-dist2 / sigma2)
        candidates.append({
            "cx": cx, "cy": cy, "area": area, "circ": circ,
            "confidence": confidence, "score": score,
        })
    return candidates


def pick_target(candidates, rule):
    """Choose the initial/reset target from candidates according to the rule."""
    if not candidates:
        return None
    if rule == "highest_conf":
        key = lambda c: c["confidence"]
    elif rule == "nearest_center":
        key = lambda c: c["score"]
    else:  # largest_blob
        key = lambda c: c["area"]
    return max(candidates, key=key)


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(Config(), args)

    source = args.input or cfg.frames_dir
    if not os.path.exists(source):
        raise FileNotFoundError(f"Input does not exist: {source}")
    output_csv = args.output or cfg.output_csv
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # 1. Read all grayscale frames (image dirs are prefetched in parallel)
    print(f"Reading frame sequence: {source}")
    frames, times = [], []
    for _, time_s, gray in iter_frames(source, cfg.fps):
        frames.append(gray)
        times.append(time_s)
    if not frames:
        raise FileNotFoundError(f"No frames read from {source}")

    # 2. Median background modeling
    master_bg = create_median_background(frames, cfg.num_bg_samples)
    master_bg = cv2.GaussianBlur(master_bg, cfg.blur_size, 0)
    if not args.no_bg_debug:
        bg_debug = os.path.join(os.path.dirname(os.path.abspath(output_csv)), "debug_master_background.png")
        cv2.imwrite(bg_debug, master_bg)
        print(f"Background model saved: {bg_debug}")

    # 3. Per-frame subtraction + detection + gated tracking
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # hoisted out of the loop
    height, width = master_bg.shape

    track_state = "UNINITIALIZED"
    missed = 0
    last_pos = None
    velocity_s = (0.0, 0.0)  # EMA-smoothed velocity
    init_stable = 0
    rows = []

    for frame_idx, (time_s, gray) in enumerate(zip(times, frames)):
        gray = cv2.GaussianBlur(gray, cfg.blur_size, 0)
        diff = cv2.absdiff(gray, master_bg)
        _, bw = cv2.threshold(diff, cfg.threshold_val, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        candidates = detect_candidates(bw, cfg.min_area, cfg.max_area, cfg.circularity_min, width, height)

        chosen, status = None, "no_detect"
        if track_state == "UNINITIALIZED":
            target = pick_target(candidates, cfg.reinit_rule)
            if target is not None:
                track_state = "TRACKING"
                missed = 0
                last_pos = (target["cx"], target["cy"])
                velocity_s = (0.0, 0.0)
                init_stable = 0
                chosen, status = target, "ok_init"
        else:
            speed = math.hypot(*velocity_s)
            gate = max(cfg.gate_radius_min_px, 8.0 + 0.8 * speed)  # adaptive gate
            search_gate = gate * (1.5 if 0 < missed <= 5 else 1.0)  # widen search on brief misses
            predicted = (last_pos[0] + velocity_s[0], last_pos[1] + velocity_s[1])
            valid = [c for c in candidates
                     if get_distance((c["cx"], c["cy"]), predicted) <= search_gate]
            if valid:
                chosen = min(valid, key=lambda c: get_distance((c["cx"], c["cy"]), predicted))
                measured = (chosen["cx"] - last_pos[0], chosen["cy"] - last_pos[1])
                if init_stable < 3:  # use raw velocity for the first frames so it converges
                    velocity_s = measured
                    init_stable += 1
                else:
                    a = cfg.ema_alpha
                    velocity_s = (a * measured[0] + (1 - a) * velocity_s[0],
                                  a * measured[1] + (1 - a) * velocity_s[1])
                last_pos = (chosen["cx"], chosen["cy"])
                missed = 0
                status = "ok_track"
            else:
                missed += 1
                decay = 0.9 ** missed  # decay velocity so extrapolation doesn't drift far after long misses
                velocity_s = (velocity_s[0] * decay, velocity_s[1] * decay)
                last_pos = predicted
                status = "lost"
                if missed > cfg.max_missed_frames:
                    track_state = "UNINITIALIZED"

        if chosen is None:
            rows.append([frame_idx, time_s, None, None, None, None, 0.0, status])
        else:
            rows.append([frame_idx, time_s, chosen["cx"], chosen["cy"],
                         chosen["area"], chosen["circ"], chosen["confidence"], status])

    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "time_s", "x_px", "y_px", "area_px2", "circularity", "confidence", "status"])
        writer.writerows(rows)

    print(f"Tracking finished: {len(rows)} frames, results saved to {output_csv}")


if __name__ == "__main__":
    main()
