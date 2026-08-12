"""Debug visualization: produces a 2x2 comparison of original/background/diff/binary per frame.

For manually checking background subtraction and blob detection results.

Usage example:
    python diff_bg_frame.py --input frames_sample18/ --output-dir diff_bg_frames/
"""
import argparse
import os

import cv2
import numpy as np

from config import Config, apply_overrides, build_common_parser
from utils import create_median_background, iter_frames


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[build_common_parser()],
        description="Generate a 2x2 debug comparison of background subtraction",
    )
    parser.add_argument("--bg-samples", type=int, default=None, dest="num_bg_samples",
                        help="Number of frames sampled for background modeling")
    parser.add_argument("--threshold", type=int, default=None, dest="threshold_val",
                        help="Binary mask threshold")
    parser.add_argument("--gain", type=float, default=5.0, help="Visualization gain for the diff")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(Config(), args)

    source = args.input or cfg.frames_dir
    if not os.path.exists(source):
        raise FileNotFoundError(f"Input does not exist: {source}")
    output_dir = args.output_dir or "diff_bg_frames"
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for _, _, gray in iter_frames(source, cfg.fps):
        frames.append(gray)
    if not frames:
        raise FileNotFoundError(f"No frames read from {source}")

    master_bg = create_median_background(frames, cfg.num_bg_samples)
    master_bg_blurred = cv2.GaussianBlur(master_bg, cfg.blur_size, 0)
    print(f"Background model built from {len(frames)} frames")

    for i, gray in enumerate(frames):
        gray_blurred = cv2.GaussianBlur(gray, cfg.blur_size, 0)
        diff = cv2.absdiff(gray_blurred, master_bg_blurred)
        visible_diff = cv2.multiply(diff, args.gain)  # amplify for easier visual inspection
        _, binary = cv2.threshold(diff, cfg.threshold_val, 255, cv2.THRESH_BINARY)

        top = np.hstack((gray, master_bg))
        bottom = np.hstack((visible_diff, binary))
        combined = np.vstack((top, bottom))
        final_view = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)

        cv2.imwrite(os.path.join(output_dir, f"cmp_{i:05d}.png"), final_view)
        if i % 100 == 0:
            print(f"Processed {i} / {len(frames)}")

    print(f"Done! Comparison images saved in '{output_dir}'")


if __name__ == "__main__":
    main()
