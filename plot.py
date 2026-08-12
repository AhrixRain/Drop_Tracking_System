"""Plotting and kinematic analysis, based on the coordinate CSV output by detect.py.

--mode trajectory        Scatter plot of the drop trajectory (colored by time, filters
                         low-confidence/lost points)
--mode height_velocity   Height/velocity/acceleration plots and a height_velocity CSV for calc_uci

Usage examples:
    python plot.py --input drop_pixel_coords.csv --mode trajectory --confidence 0.3
    python plot.py --input drop_pixel_coords.csv --mode height_velocity
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # allow saving figures without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import Config, apply_overrides, build_common_parser
from utils import pixel_to_height


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[build_common_parser()],
        description="Plot oil-drop trajectory or height/velocity/acceleration curves",
    )
    parser.add_argument("--mode", choices=("trajectory", "height_velocity"), default="trajectory")
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="Minimum confidence for trajectory mode")
    parser.add_argument("--frame-width", type=int, default=1920, help="Trajectory plot x-axis range")
    parser.add_argument("--frame-height", type=int, default=1080, help="Trajectory plot y-axis range")
    parser.add_argument("--smoothing-window", type=int, default=11, help="S-G smoothing window (must be odd)")
    parser.add_argument("--poly-order", type=int, default=3, help="S-G polynomial order")
    parser.add_argument("--moving-avg-window", type=int, default=10, help="Secondary moving-average window")
    parser.add_argument("--no-moving-avg", action="store_true", help="Disable the secondary moving average")
    return parser


def plot_trajectory(df, args, output_path):
    filtered = df[
        (df["status"] != "lost")
        & df["x_px"].notna()
        & df["y_px"].notna()
        & (df["confidence"] >= args.confidence)
    ]
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(
        filtered["x_px"], filtered["y_px"],
        c=filtered["time_s"], cmap="viridis", s=15, alpha=0.8,
    )
    plt.xlim(0, args.frame_width)
    plt.ylim(args.frame_height, 0)  # invert y so the top is 0
    plt.title(f"Millikan Oil Drop Track (Confidence >= {args.confidence})")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.grid(True, linestyle=":", alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time (seconds)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    rejected = len(df[df["status"] != "lost"]) - len(filtered)
    print(f"Trajectory plot saved: {output_path} (filtered out {rejected} points below confidence {args.confidence})")


def plot_height_velocity(df, args, cfg, output_csv, output_plot):
    """Segment by time, convert pixels to height, and use S-G smoothing derivatives for v/a."""
    df = df.dropna(subset=["y_px"]).copy()
    df = df.sort_values("time_s").reset_index(drop=True)

    # Segment by time gaps (not just frame_idx breaks); a time jump over 2 median steps starts a new segment
    median_dt = np.median(np.diff(df["time_s"].values)) if len(df) > 1 else 0.0
    segment_break = max(median_dt * 2.0, 1e-9)
    df["segment_id"] = (df["time_s"].diff() > segment_break).cumsum()
    df["height_m"] = pixel_to_height(
        df["y_px"].values, cfg.ceiling_y_px, cfg.floor_y_px, cfg.plate_spacing_m
    )

    processed = []
    for _, group in df.groupby("segment_id"):
        group = group.copy()
        h, t = group["height_m"].values, group["time_s"].values
        if len(h) < args.smoothing_window:
            group["velocity_mps"] = np.nan
            group["accel_mps2"] = np.nan
            processed.append(group)
            continue
        dt = np.median(np.diff(t))
        group["height_m"] = savgol_filter(h, args.smoothing_window, args.poly_order)
        velocity = savgol_filter(h, args.smoothing_window, args.poly_order, deriv=1, delta=dt)
        accel = savgol_filter(h, args.smoothing_window, args.poly_order, deriv=2, delta=dt)
        if not args.no_moving_avg:
            velocity = pd.Series(velocity).rolling(args.moving_avg_window, center=True).mean().values
            accel = pd.Series(accel).rolling(args.moving_avg_window, center=True).mean().values
        group["velocity_mps"] = velocity
        group["accel_mps2"] = accel
        processed.append(group)

    final = pd.concat(processed)
    final.to_csv(output_csv, index=False)

    plot_df = final.dropna(subset=["velocity_mps", "accel_mps2"])
    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    for _, g in plot_df.groupby("segment_id"):
        plt.plot(g["time_s"], g["height_m"])
    plt.title("Height vs Time")
    plt.ylabel("Height (m)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    for _, g in plot_df.groupby("segment_id"):
        plt.plot(g["time_s"], g["velocity_mps"], color="darkorange")
    plt.title("Velocity vs Time")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    for _, g in plot_df.groupby("segment_id"):
        plt.plot(g["time_s"], g["accel_mps2"], color="crimson")
    plt.title("Accel vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel (m/s²)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Height/velocity/acceleration data saved: {output_csv}")
    print(f"Plots saved: {output_plot}")


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(Config(), args)

    input_csv = args.input or cfg.output_csv
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cannot find coordinate CSV: {input_csv}")

    stem, _ = os.path.splitext(input_csv)
    if args.mode == "trajectory":
        output_path = args.output or f"{stem}_track_plot.png"
        plot_trajectory(pd.read_csv(input_csv), args, output_path)
    else:
        output_csv = args.output or f"{stem}_height_velocity.csv"
        output_plot = f"{os.path.splitext(output_csv)[0]}_plots.png"
        plot_height_velocity(pd.read_csv(input_csv), args, cfg, output_csv, output_plot)


if __name__ == "__main__":
    main()
