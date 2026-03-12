import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

# ==========================================
# CONFIGURATION
# Change the name according to your csv name
# ==========================================
pixel_table_csv = "drop_pixel_coords_sample18.csv"
output_csv = "height_velocity_sample18.csv"

# Calibration Params
# You should manually determine this for every setup
ceiling_y_px = 200.0       
floor_y_px = 1000.0         
separation_L = 0.003       

# AVERAGING PARAMETERS
# Increase window_length for more smoothing (must be ODD)
# Increase moving_average_window for flatter lines
smoothing_window = 11      # Default was 11; try 31, 51, or 71
poly_order = 3             # Higher order (3) helps preserve the actual 'curve'
apply_moving_avg = True    # Set to True for extra smoothness
moving_avg_window = 10     # Number of points for the rolling mean

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(pixel_table_csv):
        print(f"Error: Could not find {pixel_table_csv}")
        return

    df = pd.read_csv(pixel_table_csv)
    df = df.sort_values(by="time_s").reset_index(drop=True)

    # Handle Segments
    df = df.dropna(subset=['y_px']).copy()
    df['frame_diff'] = df['frame_idx'].diff()
    df['segment_id'] = (df['frame_diff'] > 1).cumsum()

    # Calibration
    px_span = floor_y_px - ceiling_y_px
    df['height_m'] = separation_L * (floor_y_px - df['y_px']) / px_span

    processed_segments = []

    for segment_id, group in df.groupby('segment_id'):
        group = group.copy()
        t = group['time_s'].values
        h = group['height_m'].values
        
        # Skip segments too short for the window
        if len(h) < smoothing_window:
            group['velocity_mps'] = np.nan
            group['accel_mps2'] = np.nan
            processed_segments.append(group)
            continue
            
        dt = np.median(np.diff(t))

        # 1. Savitzky-Golay Smoothing & Derivatives
        # This performs a local polynomial fit
        group['height_m'] = savgol_filter(h, window_length=smoothing_window, polyorder=poly_order)
        v = savgol_filter(h, window_length=smoothing_window, polyorder=poly_order, deriv=1, delta=dt)
        a = savgol_filter(h, window_length=smoothing_window, polyorder=poly_order, deriv=2, delta=dt)

        # 2. Secondary Rolling Average (The 'Averaging' step)
        if apply_moving_avg:
            # We use a pandas Series to easily apply a centered rolling mean
            v = pd.Series(v).rolling(window=moving_avg_window, center=True).mean().values
            a = pd.Series(a).rolling(window=moving_avg_window, center=True).mean().values
            
        group['velocity_mps'] = v
        group['accel_mps2'] = a
        processed_segments.append(group)

    # Recombine segments
    df_final = pd.concat(processed_segments)
    df_final.to_csv(output_csv, index=False)

    # --- Plotting ---
    plot_df = df_final.dropna(subset=['velocity_mps', 'accel_mps2'])
    plt.figure(figsize=(10, 12))

    # Height
    plt.subplot(3, 1, 1)
    for sid, g in plot_df.groupby('segment_id'):
        plt.plot(g['time_s'], g['height_m'])
    plt.title(f'Height vs Time')
    plt.ylabel('Height (m)')
    plt.grid(True, alpha=0.3)

    # Velocity
    plt.subplot(3, 1, 2)
    for sid, g in plot_df.groupby('segment_id'):
        plt.plot(g['time_s'], g['velocity_mps'], color='darkorange')
    plt.title('Velocity vs Time')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True, alpha=0.3)

    # Acceleration
    plt.subplot(3, 1, 3)
    for sid, g in plot_df.groupby('segment_id'):
        plt.plot(g['time_s'], g['accel_mps2'], color='crimson')
    plt.title('Accel vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel (m/s²)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('height_velocity_accel_plots_sample18.png', dpi=300)
    print("Averaged plots saved.")

if __name__ == "__main__":
    main()