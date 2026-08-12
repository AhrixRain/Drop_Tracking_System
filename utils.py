"""共享工具：背景建模、帧来源迭代、像素标定换算。"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def create_median_background(frames, num_samples):
    """从帧序列中均匀采样 num_samples 帧，取中位数合成静态背景。

    中位数优于均值：可剔除序列中移动的油滴，避免背景被污染。
    """
    num = min(num_samples, len(frames))
    if num <= 0:
        raise ValueError("没有足够的帧来构建背景模型")
    indices = np.linspace(0, len(frames) - 1, num, dtype=int)
    stack = np.stack([frames[i] for i in indices]).astype(np.uint8)
    return np.median(stack, axis=0).astype(np.uint8)


def iter_frames(source, fps=None):
    """遍历帧来源，产出 (frame_idx, time_s, gray_frame)。

    支持两种来源：图片帧目录（按文件名排序）或单个视频文件。
    图片目录用线程池并行预读灰度图（imread 是 C 调用，多线程可提升 IO 吞吐）；
    视频用 cv2.VideoCapture，时间戳由视频真实帧率推算。
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
        raise FileNotFoundError(f"在 {source} 中未找到图片帧")

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
        raise FileNotFoundError(f"无法打开视频: {source}")
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
    """把像素 y 坐标换算为相对顶部（ceiling）的高度，单位米。"""
    if floor_y_px == ceiling_y_px:
        raise ValueError("--floor-y-px 与 --ceiling-y-px 不能相等")
    return plate_spacing_m * (floor_y_px - y_px) / (floor_y_px - ceiling_y_px)
