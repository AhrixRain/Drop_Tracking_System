import cv2
import os
import glob
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
input_dir = "frames/"
output_dir = "diff_bg_frames/"
frame_pattern = "*.png"  # Change to *.jpg if necessary

# Background Sampling
num_bg_samples = 30  # Number of frames to average/median to create background

# Visualization Tweaks
visibility_gain = 5.0 # Boost the difference so you can see faint drops
threshold_val = 25    # The threshold used to create the binary mask

# ==========================================
# EXECUTION
# ==========================================
def create_median_bg(files, samples):
    print(f"Building master background from {samples} frames...")
    indices = np.linspace(0, len(files) - 1, samples, dtype=int)
    batch = []
    for idx in indices:
        f = cv2.imread(files[idx], cv2.IMREAD_GRAYSCALE)
        if f is not None:
            batch.append(f)
    # Median is better than mean for removing moving drops from the reference
    return np.median(batch, axis=0).astype(np.uint8)

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted(glob.glob(os.path.join(input_dir, frame_pattern)))
    if not files:
        print("No frames found!")
        return

    # 1. Generate the static reference
    master_bg = create_median_bg(files, num_bg_samples)
    master_bg_blurred = cv2.GaussianBlur(master_bg, (5, 5), 0)

    print(f"Generating comparison frames in '{output_dir}'...")

    for i, file_path in enumerate(files):
        img = cv2.imread(file_path)
        if img is None: continue
        
        # Convert to gray and blur (standardizing the input)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Calculate the difference from the Master Background
        diff = cv2.absdiff(gray_blurred, master_bg_blurred)

        # 3. Apply gain for human eyes
        visible_diff = cv2.multiply(diff, visibility_gain)

        # 4. Create the binary mask (the "Detection" view)
        _, binary = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

        # 5. Stack the views side-by-side
        # Panel 1: Raw Gray
        # Panel 2: The static Background model
        # Panel 3: The Difference (multiplied by gain)
        # Panel 4: What the computer "sees" as a blob
        top_row = np.hstack((gray, master_bg))
        bottom_row = np.hstack((visible_diff, binary))
        
        # Combining them into a single image (scaled down so it fits on screen)
        combined = np.vstack((top_row, bottom_row))
        final_view = cv2.resize(combined, (0,0), fx=0.5, fy=0.5)

        # Save result
        fname = os.path.basename(file_path)
        cv2.imwrite(os.path.join(output_dir, f"cmp_{fname}"), final_view)

        if i % 100 == 0:
            print(f"Processed {i} / {len(files)}...")

    print(f"\nDone! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    main()