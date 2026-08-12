"""Central configuration: all constants live here; scripts override defaults via CLI args."""
import argparse
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    # ---- Physical constants ----
    g_m_s2: float = 9.81
    r_specific_air: float = 287.058  # J/(kg·K)
    elementary_charge_c: float = 1.602176634e-19
    # Cunningham correction b, from UCI notes 5.908e-3 torr-cm, converted to Pa·m
    cunningham_b_pa_m: float = 5.908e-3 * 133.322 * 1e-2

    # ---- Calibration (set per actual experimental setup) ----
    ceiling_y_px: float = 200.0
    floor_y_px: float = 1000.0
    plate_spacing_m: float = 0.003175

    # ---- Detection / tracking ----
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

    # ---- Default paths ----
    frames_dir: str = "frames"
    output_csv: str = "drop_pixel_coords.csv"


def build_common_parser() -> argparse.ArgumentParser:
    """Common CLI args shared by all scripts, reused via argparse parents=[...]."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input", default=None, help="Input: frame directory or video file path")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--fps", type=float, default=None, help="Frame rate (frame-dir mode infers time from this)")
    parser.add_argument("--ceiling-y-px", type=float, default=None, help="Calibrated top y pixel")
    parser.add_argument("--floor-y-px", type=float, default=None, help="Calibrated bottom y pixel")
    parser.add_argument("--plate-spacing", type=float, default=None, dest="plate_spacing_m",
                        help="Plate spacing (m), also the pixel-calibration height span")
    return parser


def apply_overrides(cfg: Config, args) -> Config:
    """Override Config fields with same-named values from an argparse Namespace."""
    values = {}
    for field_name in Config.__dataclass_fields__:
        value = getattr(args, field_name, None)
        if value is not None:
            values[field_name] = value
    return replace(cfg, **values)
