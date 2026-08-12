# Drop_Tracking_System

Automatic tracking and analysis system for the Millikan oil-drop experiment (Physics 121W): tracks a single oil drop from video frames, performs height/velocity/acceleration analysis, and finally computes the drop's charge following the UCI notes method.

## Pipeline

```
Video / frame directory
   │
   ├─ detect.py          Oil-drop detection and tracking → drop_pixel_coords.csv (per-frame coords)
   │
   ├─ diff_bg_frame.py   Debug visualization: original/background/diff/binary 2x2 comparison
   │
   ├─ plot.py
   │   ├─ --mode trajectory        trajectory scatter plot (colored by time)
   │   └─ --mode height_velocity   height/velocity/acceleration plots + height_velocity CSV
   │
   └─ calc_uci.py        Charge calculation → *_calculated_results_uci.csv (includes n_e)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Detection and tracking (video or frame directory input)

```bash
python detect.py --input frames_sample18/ --output drop_pixel_coords.csv
python detect.py --input movie.mp4 --output drop_pixel_coords.csv --fps 60
```

Optional flags: `--bg-samples` (background sampling frames), `--threshold` (sensitivity), `--min-area/--max-area/--circularity` (blob filtering), `--gate-radius/--max-missed/--ema-alpha` (tracking), `--reinit-rule {largest_blob,highest_conf,nearest_center}`.

### 2. Debug visualization (optional)

```bash
python diff_bg_frame.py --input frames_sample18/ --output-dir diff_bg_frames/
```

### 3. Plotting and kinematic analysis

```bash
# Trajectory scatter plot
python plot.py --input drop_pixel_coords.csv --mode trajectory --confidence 0.3
# Height/velocity/acceleration (generates height_velocity CSV)
python plot.py --input drop_pixel_coords.csv --mode height_velocity
```

### 4. Charge calculation

```bash
python calc_uci.py --input height_velocity_sample18.csv        # reads velocity directly
python calc_uci.py --input drop_pixel_coords.csv --method free_fall
```

See `python calc_uci.py --help` for more options (voltage, plate spacing, temperature, pressure, oil density, calibration pixels, etc. are all overridable).

## Calibration notes

Default calibration parameters for all scripts are centralized in `config.py` and can be overridden via CLI:

- `--ceiling-y-px / --floor-y-px`: top/bottom y pixels corresponding to the calibrated height range in the image.
- `--plate-spacing`: plate spacing (m), also used as the pixel-calibration height span.

**Be sure to set these values to match your actual experimental setup**, otherwise the height and charge results will be skewed.
