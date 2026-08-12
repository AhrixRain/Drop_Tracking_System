"""统一配置：全部常量集中于此，各脚本通过 CLI 参数覆盖默认值。"""
import argparse
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # ---- 物理常数 ----
    g_m_s2: float = 9.81
    r_specific_air: float = 287.058  # J/(kg·K)
    elementary_charge_c: float = 1.602176634e-19
    # Cunningham 修正 b，来自 UCI 讲义 5.908e-3 torr-cm，换算为 Pa·m
    cunningham_b_pa_m: float = 5.908e-3 * 133.322 * 1e-2

    # ---- 标定（每个实验装置需按实际测量设置）----
    ceiling_y_px: float = 200.0
    floor_y_px: float = 1000.0
    plate_spacing_m: float = 0.003175

    # ---- 检测 / 跟踪 ----
    fps: float = 60.0
    num_bg_samples: int = 30
    blur_size: tuple[int, int] = (5, 5)
    threshold_val: int = 25
    min_area: int = 4
    max_area: int = 500
    circularity_min: float = 0.1
    gate_radius_min_px: float = 8.0
    max_missed_frames: int = 10
    ema_alpha: float = 0.4
    reinit_rule: str = "largest_blob"  # largest_blob | highest_conf | nearest_center

    # ---- 默认路径 ----
    frames_dir: str = "frames"
    output_csv: str = "drop_pixel_coords.csv"


def build_common_parser() -> argparse.ArgumentParser:
    """各脚本共享的通用参数，用 argparse parents=[...] 复用。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=None, help="输入：帧目录或视频文件路径")
    parser.add_argument("--output", default=None, help="输出文件路径")
    parser.add_argument("--fps", type=float, default=None, help="帧率（帧目录模式用于推算时间）")
    parser.add_argument("--ceiling-y-px", type=float, default=None, help="标定顶部 y 像素")
    parser.add_argument("--floor-y-px", type=float, default=None, help="标定底部 y 像素")
    parser.add_argument("--plate-spacing", type=float, default=None, dest="plate_spacing_m",
                        help="极板间距（米），兼作像素标定高度跨度")
    return parser


def apply_overrides(cfg: Config, args) -> Config:
    """用 argparse 的 Namespace 覆盖 Config 中同名字段的默认值。"""
    values = {}
    for field_name in Config.__dataclass_fields__:
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    return replace(cfg, **values)
