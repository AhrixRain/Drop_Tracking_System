"""Shared utilities: background modeling, frame-source iteration, pixel calibration."""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def create_median_background(frames, num_samples):
    """Uniformly sample num_samples frames and synthesize a static background by median.

    Median beats mean: it rejects moving oil drops in the sequence, keeping the background clean.
    """
    num = min(num_samples, len(frames))
    if num <= 0:
        raise ValueError("Not enough frames to build a background model")
    indices = np.linspace(0, len(frames) - 1, num, dtype=int)
    stack = np.stack([frames[i] for i in indices]).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def iter_frames(source, fps=None):
    """Iterate a frame source, yielding (frame_idx, time_s, gray_frame).

    Supports two sources: an image frame directory (sorted by filename) or a single video file.
    Image directories prefetch grayscale frames in parallel via a thread pool (imread is a C
    call, so threads improve IO throughput); videos use cv2.VideoCapture with timestamps derived
    from the video's actual frame rate.
    """
    if os.path.isdir(source):
        yield from _iter_image_dir(source, fps)
    else:
        yield from _iter_video(source, fps)


def _iter_image_dir(source, fps):
    files = sorted(
        f for f in glob.glob(os.path.join(source, "*"))
        if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No image frames found in {source}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(cv2.imread, f, cv2.IMREAD_GRAYSCALE): f for f in files}
        for idx, future in enumerate(future_map):
            img = future.result()
            if img is None:
                continue
            yield idx, (idx / fps if fps else 0.0), img


def _iter_video(source, fps):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 25.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, idx / video_fps, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        idx += 1
    cap.release()


def pixel_to_height(y_px, ceiling_y_px, floor_y_px, plate_spacing_m):
    """Convert pixel y to height in meters relative to the top (ceiling)."""
    if floor_y_px == ceiling_y_px:
        raise ValueError("--floor-y-px and --ceiling-y-px cannot be equal")
    return plate_spacing_m * (floor_y_px - y_px) / (floor_y_px - ceiling_y_px)
