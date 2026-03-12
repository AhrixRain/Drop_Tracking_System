import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration
csv_path = 'drop_pixel_coords_sample18.csv'
output_image = 'filtered_drop_track_plot_sample18.png'
confidence_threshold = 0.  # Adjust this value as needed

# 2. Load and Filter Data
df = pd.read_csv(csv_path)

# Filter logic: Not lost AND has coordinates AND meets confidence threshold
filtered_df = df[
    (df['status'] != 'lost') & 
    (df['x_px'].notna()) & 
    (df['y_px'].notna()) & 
    (df['confidence'] >= confidence_threshold)
]

# 3. Visualization
plt.figure(figsize=(12, 7))

# Scatter plot colored by time
scatter = plt.scatter(
    filtered_df['x_px'], 
    filtered_df['y_px'], 
    c=filtered_df['time_s'], 
    cmap='viridis', 
    s=15, 
    alpha=0.8
)

# Set axes to match 1920x1080 image resolution
plt.xlim(0, 1920)
plt.ylim(1080, 0) # Invert Y so 0 is at the top

plt.title(f'Millikan Oil Drop Track (Confidence >= {confidence_threshold})')
plt.xlabel('X (px)')
plt.ylabel('Y (px)')
plt.grid(True, linestyle=':', alpha=0.6)

# Add colorbar for time
cbar = plt.colorbar(scatter)
cbar.set_label('Time (seconds)')

plt.savefig(output_image, dpi=300, bbox_inches='tight')

# Console output for verification
rejected_count = len(df[df['status'] != 'lost']) - len(filtered_df)
print(f"Plot saved. Rejected {rejected_count} points below {confidence_threshold} confidence.")