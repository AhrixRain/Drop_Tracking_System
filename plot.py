"""绘图与运动学分析：基于 detect.py 输出的坐标 CSV。

--mode trajectory       油滴轨迹散点图（按时间着色，可过滤低置信度/丢失点）
--mode height_velocity  高度/速度/加速度图，并输出 height_velocity CSV 供 calc_uci 使用

用法示例：
    python plot.py --input drop_pixel_coords.csv --mode trajectory --confidence 0.3
    python plot.py --input drop_pixel_coords.csv --mode height_velocity
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import Config, apply_overrides, build_common_parser
from utils import pixel_to_height


def build_parser():
    parser = argparse.ArgumentParser(
        parents=[build_common_parser()],
        description="绘制油滴轨迹或高度/速度/加速度曲线",
    )
    parser.add_argument("--mode", choices=("trajectory", "height_velocity"), default="trajectory")
    parser.add_argument("--confidence", type=float, default=0.0,
                        help="trajectory 模式的最低置信度")
    parser.add_argument("--frame-width", type=int, default=1920, help="轨迹图 x 轴范围")
    parser.add_argument("--frame-height", type=int, default=1080, help="轨迹图 y 轴范围")
    parser.add_argument("--smoothing-window", type=int, default=11, help="S-G 平滑窗口（须为奇数）")
    parser.add_argument("--poly-order", type=int, default=3, help="S-G 多项式阶数")
    parser.add_argument("--moving-avg-window", type=int, default=10, help="二次滑动平均窗口")
    parser.add_argument("--no-moving-avg", action="store_true", help="禁用二次滑动平均")
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
    plt.ylim(args.frame_height, 0)  # 反转 y，使顶部为 0
    plt.title(f"Millikan Oil Drop Track (Confidence >= {args.confidence})")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.grid(True, linestyle=":", alpha=0.6)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Time (seconds)")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    rejected = len(df[df["status"] != "lost"]) - len(filtered)
    print(f"轨迹图已保存: {output_path}（按置信度 {args.confidence} 过滤掉 {rejected} 点）")


def plot_height_velocity(df, args, cfg, output_csv, output_plot):
    """按时间分段，把像素坐标换算为高度，用 S-G 平滑求导得到速度/加速度。"""
    df = df.dropna(subset=["y_px"]).copy()
    df = df.sort_values("time_s").reset_index(drop=True)

    # 分段：按时间间隔（而非仅 frame_idx 断点），时间跳变超过 2 个中位步长视为新段
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
    print(f"高度/速度/加速度数据已保存: {output_csv}")
    print(f"曲线图已保存: {output_plot}")


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(Config(), args)

    input_csv = args.input or cfg.output_csv
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"找不到坐标 CSV: {input_csv}")

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
