import argparse
import csv
import math
import os
from statistics import mean, stdev


# ==========================================
# DEFAULT CONFIGURATION
# CLI flags can override any of these.
# ==========================================
INPUT_CSV = "drop_pixel_coords/drop_pixel_coords_sample12.csv"
METHOD = "free_fall"  # "free_fall" or "forced_down"
VOLTAGE_V = 300.0
PLATE_SPACING_M = 0.003175
TEMPERATURE_C = 23.0
PRESSURE_TORR = 757.0
RHO_OIL_KG_M3 = 800.0
VELOCITY_THRESHOLD_MPS = 5e-5
COORD_CEILING_Y_PX = 200.0
COORD_FLOOR_Y_PX = 1000.0
COORD_HEIGHT_SPAN_M = 0.003175
DERIVATIVE_WINDOW = 11

G_M_S2 = 9.81
R_SPECIFIC_AIR = 287.058
ELEMENTARY_CHARGE_C = 1.602176634e-19

# UCI handout constant: b = 5.908e-3 torr-cm
# Converted to SI: Pa*m
CUNNINGHAM_B_PA_M = 5.908e-3 * 133.322 * 1e-2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calculate Millikan oil-drop charge using the UCI handout method."
    )
    parser.add_argument(
        "--input",
        default=INPUT_CSV,
        help="Input coord CSV or prepared height/velocity CSV",
    )
    parser.add_argument(
        "--method",
        choices=("free_fall", "forced_down"),
        default=METHOD,
        help="Use Eq. 14 (free_fall) or Eq. 15 (forced_down).",
    )
    parser.add_argument("--voltage", type=float, default=VOLTAGE_V, help="Plate voltage in volts")
    parser.add_argument(
        "--plate-spacing",
        type=float,
        default=PLATE_SPACING_M,
        help="Plate spacing in meters",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE_C,
        help="Room temperature in Celsius",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=PRESSURE_TORR,
        help="Atmospheric pressure in torr (mmHg)",
    )
    parser.add_argument(
        "--rho-oil",
        type=float,
        default=RHO_OIL_KG_M3,
        help="Oil density in kg/m^3",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=VELOCITY_THRESHOLD_MPS,
        help="Minimum |velocity| used to classify rising/falling terminal data",
    )
    parser.add_argument(
        "--ceiling-y-px",
        type=float,
        default=COORD_CEILING_Y_PX,
        help="Top calibration y pixel used to convert coord CSV rows into height",
    )
    parser.add_argument(
        "--floor-y-px",
        type=float,
        default=COORD_FLOOR_Y_PX,
        help="Bottom calibration y pixel used to convert coord CSV rows into height",
    )
    parser.add_argument(
        "--coord-height-span",
        type=float,
        default=COORD_HEIGHT_SPAN_M,
        help="Physical height in meters between the ceiling/floor calibration pixels",
    )
    parser.add_argument(
        "--derivative-window",
        type=int,
        default=DERIVATIVE_WINDOW,
        help="Odd window size for local linear-fit velocity estimation from coord CSV rows",
    )
    return parser


def parse_float(raw_value):
    if raw_value in ("", None, "nan", "NaN"):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def parse_int(raw_value):
    if raw_value in ("", None, "nan", "NaN"):
        return None
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return None


def regression_slope(xs, ys):
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        return None
    return numerator / denominator


def heights_to_velocities(times, heights, window):
    if len(times) < window:
        return []

    half_window = window // 2
    velocities = []
    for center_idx in range(half_window, len(times) - half_window):
        time_window = times[center_idx - half_window : center_idx + half_window + 1]
        height_window = heights[center_idx - half_window : center_idx + half_window + 1]
        slope = regression_slope(time_window, height_window)
        if slope is not None:
            velocities.append(slope)
    return velocities


def read_coord_velocities(path, ceiling_y_px, floor_y_px, coord_height_span_m, derivative_window):
    if derivative_window < 3 or derivative_window % 2 == 0:
        raise ValueError("--derivative-window must be an odd integer >= 3")
    if floor_y_px == ceiling_y_px:
        raise ValueError("--floor-y-px and --ceiling-y-px must be different")
    if coord_height_span_m <= 0:
        raise ValueError("--coord-height-span must be positive")

    px_span = floor_y_px - ceiling_y_px
    points = []

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "time_s" not in fieldnames or "y_px" not in fieldnames:
            raise ValueError(f"{path} must contain time_s and y_px columns")

        for row in reader:
            time_s = parse_float(row.get("time_s"))
            y_px = parse_float(row.get("y_px"))
            if time_s is None or y_px is None:
                continue

            height_m = coord_height_span_m * (floor_y_px - y_px) / px_span
            points.append(
                {
                    "frame_idx": parse_int(row.get("frame_idx")),
                    "time_s": time_s,
                    "height_m": height_m,
                }
            )

    if not points:
        raise ValueError(f"{path} does not contain usable time_s/y_px coord rows")

    points.sort(key=lambda row: row["time_s"])

    segments = []
    current_segment = [points[0]]
    for point in points[1:]:
        previous = current_segment[-1]
        frame_gap = None
        if point["frame_idx"] is not None and previous["frame_idx"] is not None:
            frame_gap = point["frame_idx"] - previous["frame_idx"]

        if point["time_s"] <= previous["time_s"] or (frame_gap is not None and frame_gap > 1):
            segments.append(current_segment)
            current_segment = [point]
            continue

        current_segment.append(point)
    segments.append(current_segment)

    velocities = []
    for segment in segments:
        if len(segment) < derivative_window:
            continue
        times = [row["time_s"] for row in segment]
        heights = [row["height_m"] for row in segment]
        velocities.extend(heights_to_velocities(times, heights, derivative_window))

    if not velocities:
        raise ValueError(
            f"{path} does not contain enough continuous coord rows for --derivative-window={derivative_window}"
        )
    return velocities, "coord_csv"


def read_prepared_velocities(path):
    velocities = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if "velocity_mps" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing a velocity_mps column")
        for row in reader:
            velocity = parse_float(row.get("velocity_mps"))
            if velocity is not None:
                velocities.append(velocity)
    if not velocities:
        raise ValueError(f"{path} does not contain any usable velocity_mps values")
    return velocities, "velocity_csv"


def read_velocities(path, ceiling_y_px, floor_y_px, coord_height_span_m, derivative_window):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

    if {"time_s", "y_px"}.issubset(fieldnames):
        return read_coord_velocities(
            path,
            ceiling_y_px=ceiling_y_px,
            floor_y_px=floor_y_px,
            coord_height_span_m=coord_height_span_m,
            derivative_window=derivative_window,
        )
    if "velocity_mps" in fieldnames:
        return read_prepared_velocities(path)
    raise ValueError(f"{path} is missing the columns needed to derive velocities")


def air_density_kg_m3(temperature_c, pressure_torr):
    temperature_k = temperature_c + 273.15
    pressure_pa = pressure_torr * 133.322
    return pressure_pa / (R_SPECIFIC_AIR * temperature_k)


def eta0_pa_s(temperature_c):
    temp_ratio = (273.16 + temperature_c) / 293.16
    sutherland_ratio = (293.16 + 110.4) / (273.16 + temperature_c + 110.4)
    # Eq. 10 from the UCI handout, converted from poise to Pa*s.
    return 1.81804e-5 * (temp_ratio ** 1.5) * sutherland_ratio


def positive_root(quadratic_c, pressure_pa):
    linear_term = CUNNINGHAM_B_PA_M / pressure_pa
    return math.sqrt((linear_term / 2.0) ** 2 + quadratic_c) - (linear_term / 2.0)


def standard_error(values):
    if len(values) < 2:
        return 0.0
    return stdev(values) / math.sqrt(len(values))


def summarize_velocities(velocities, threshold):
    rising = [v for v in velocities if v > threshold]
    falling = [-v for v in velocities if v < -threshold]

    if not rising:
        raise ValueError("No rising terminal-velocity points were found above the threshold")
    if not falling:
        raise ValueError("No falling terminal-velocity points were found below the threshold")

    return {
        "rising_values": rising,
        "falling_values": falling,
        "v_rise_mps": mean(rising),
        "v_fall_mps": mean(falling),
        "v_rise_sem_mps": standard_error(rising),
        "v_fall_sem_mps": standard_error(falling),
    }


def calculate_radius_uncorrected(method, eta0, rho_prime, v_fall, v_rise):
    if method == "free_fall":
        return math.sqrt((9.0 * eta0 * v_fall) / (2.0 * rho_prime * G_M_S2))

    delta_v = v_fall - v_rise
    if delta_v <= 0:
        raise ValueError("forced_down requires v_fall > v_rise to produce a real radius")
    return math.sqrt((9.0 * eta0 * delta_v) / (4.0 * rho_prime * G_M_S2))


def calculate_radius_corrected(method, eta0, pressure_pa, rho_prime, v_fall, v_rise):
    if method == "free_fall":
        c_term = (9.0 * eta0 * v_fall) / (2.0 * rho_prime * G_M_S2)
        return positive_root(c_term, pressure_pa)

    delta_v = v_fall - v_rise
    if delta_v <= 0:
        raise ValueError("forced_down requires v_fall > v_rise to produce a real radius")
    c_term = (9.0 * eta0 * delta_v) / (4.0 * rho_prime * G_M_S2)
    return positive_root(c_term, pressure_pa)


def corrected_viscosity(eta0, radius_m, pressure_pa):
    return eta0 / (1.0 + CUNNINGHAM_B_PA_M / (radius_m * pressure_pa))


def calculate_charge(method, eta, radius_m, v_fall, v_rise, electric_field_vpm):
    prefactor = 6.0 if method == "free_fall" else 3.0
    return prefactor * math.pi * eta * radius_m * (v_fall + v_rise) / electric_field_vpm


def output_path_for(input_csv):
    stem, _ = os.path.splitext(input_csv)
    return f"{stem}_calculated_results_uci.csv"


def write_results(path, row):
    fieldnames = list(row.keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    args = build_parser().parse_args()

    velocities, input_mode = read_velocities(
        args.input,
        ceiling_y_px=args.ceiling_y_px,
        floor_y_px=args.floor_y_px,
        coord_height_span_m=args.coord_height_span,
        derivative_window=args.derivative_window,
    )
    velocity_stats = summarize_velocities(velocities, args.velocity_threshold)

    v_rise = velocity_stats["v_rise_mps"]
    v_fall = velocity_stats["v_fall_mps"]

    pressure_pa = args.pressure * 133.322
    electric_field_vpm = args.voltage / args.plate_spacing
    rho_air = air_density_kg_m3(args.temperature, args.pressure)
    rho_prime = args.rho_oil - rho_air
    eta0 = eta0_pa_s(args.temperature)

    if rho_prime <= 0:
        raise ValueError("rho_oil must be greater than the air density")
    if electric_field_vpm <= 0:
        raise ValueError("voltage and plate spacing must define a positive electric field")

    radius_uncorrected = calculate_radius_uncorrected(
        args.method, eta0, rho_prime, v_fall, v_rise
    )
    charge_uncorrected = calculate_charge(
        args.method, eta0, radius_uncorrected, v_fall, v_rise, electric_field_vpm
    )

    radius_corrected = calculate_radius_corrected(
        args.method, eta0, pressure_pa, rho_prime, v_fall, v_rise
    )
    eta_corrected = corrected_viscosity(eta0, radius_corrected, pressure_pa)
    charge_corrected = calculate_charge(
        args.method, eta_corrected, radius_corrected, v_fall, v_rise, electric_field_vpm
    )

    output_csv = output_path_for(args.input)
    results = {
        "input_csv": args.input,
        "input_mode": input_mode,
        "method": args.method,
        "voltage_v": args.voltage,
        "plate_spacing_m": args.plate_spacing,
        "electric_field_v_per_m": electric_field_vpm,
        "temperature_c": args.temperature,
        "pressure_torr": args.pressure,
        "rho_oil_kg_m3": args.rho_oil,
        "rho_air_kg_m3": rho_air,
        "eta0_pa_s": eta0,
        "eta_corrected_pa_s": eta_corrected,
        "velocity_threshold_mps": args.velocity_threshold,
        "coord_ceiling_y_px": args.ceiling_y_px,
        "coord_floor_y_px": args.floor_y_px,
        "coord_height_span_m": args.coord_height_span,
        "derivative_window": args.derivative_window,
        "num_velocity_points": len(velocities),
        "num_rising_points": len(velocity_stats["rising_values"]),
        "num_falling_points": len(velocity_stats["falling_values"]),
        "mean_rising_velocity_mps": v_rise,
        "mean_falling_velocity_mps": v_fall,
        "sem_rising_velocity_mps": velocity_stats["v_rise_sem_mps"],
        "sem_falling_velocity_mps": velocity_stats["v_fall_sem_mps"],
        "radius_uncorrected_m": radius_uncorrected,
        "radius_uncorrected_um": radius_uncorrected * 1e6,
        "charge_uncorrected_c": charge_uncorrected,
        "n_uncorrected_e": charge_uncorrected / ELEMENTARY_CHARGE_C,
        "radius_corrected_m": radius_corrected,
        "radius_corrected_um": radius_corrected * 1e6,
        "charge_corrected_c": charge_corrected,
        "n_corrected_e": charge_corrected / ELEMENTARY_CHARGE_C,
    }
    write_results(output_csv, results)

    print(f"Method: {args.method}")
    print(f"Input CSV: {args.input}")
    print(f"Input mode: {input_mode}")
    print(f"Mean rising velocity : {v_rise:.6e} m/s ({len(velocity_stats['rising_values'])} points)")
    print(f"Mean falling velocity: {v_fall:.6e} m/s ({len(velocity_stats['falling_values'])} points)")
    print(f"Uncorrected radius   : {radius_uncorrected:.6e} m")
    print(f"Corrected radius     : {radius_corrected:.6e} m")
    print(f"Corrected charge     : {charge_corrected:.6e} C")
    print(f"Corrected charge / e : {charge_corrected / ELEMENTARY_CHARGE_C:.3f}")
    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()
