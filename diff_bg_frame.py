"""调试可视化：为每一帧生成 原始/背景/差分/二值 四联对比图。

用于人工检查背景差分与 blob 检测的效果。

用法示例：
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
        description="生成背景差分的四联调试对比图",
    )
    parser.add_argument("--bg-samples", type=int, default=None, dest="num_bg_samples",
                        help="背景建模采样帧数")
    parser.add_argument("--threshold", type=int, default=None, dest="threshold_val",
                        help="二值掩膜阈值")
    parser.add_argument("--gain", type=float, default=5.0, help="差分可视化增益")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = apply_overrides(Config(), args)

    source = args.input or cfg.frames_dir
    if not os.path.exists(source):
        raise FileNotFoundError(f"输入不存在: {source}")
    output_dir = args.output_dir or "diff_bg_frames"
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for _, _, gray in iter_frames(source, cfg.fps):
        frames.append(gray)
    if not frames:
        raise FileNotFoundError(f"未从 {source} 读到任何帧")

    master_bg = create_median_background(frames, cfg.num_bg_samples)
    master_bg_blurred = cv2.GaussianBlur(master_bg, cfg.blur_size, 0)
    print(f"背景模型构建完成，共 {len(frames)} 帧")

    for i, gray in enumerate(frames):
        gray_blurred = cv2.GaussianBlur(gray, cfg.blur_size, 0)
        diff = cv2.absdiff(gray_blurred, master_bg_blurred)
        visible_diff = cv2.multiply(diff, args.gain)  # 增益放大，便于人眼观察
        _, binary = cv2.threshold(diff, cfg.threshold_val, 255, cv2.THRESH_BINARY)

        top = np.hstack((gray, master_bg))
        bottom = np.hstack((visible_diff, binary))
        combined = np.vstack((top, bottom))
        final_view = cv2.resize(combined, (0, 0), fx=0.5, fy=0.5)

        cv2.imwrite(os.path.join(output_dir, f"cmp_{i:05d}.png"), final_view)
        if i % 100 == 0:
            print(f"已处理 {i} / {len(frames)}")

    print(f"完成！对比图保存在 '{output_dir}'")


if __name__ == "__main__":
    main()
