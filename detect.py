import cv2
import os
import glob
import csv
import math
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
frames_dir = "frames_sample18/"
frame_pattern = "*.jpg"
fps = 60.0

# Median Sampling: How many frames to use to build the background model?
# Increasing this makes the background cleaner but takes longer to start.
num_bg_samples = 30 

# Image Processing Params
blur_size = (5, 5)
threshold_val = 25  # Sensitivity: Lower = more sensitive to faint drops

# Blob Filtering
min_area = 4
max_area = 500
circularity_min = 0.1

# Tracking
gate_radius_px = 40.0
max_missed_frames = 10
reinit_rule = "largest_blob"

output_csv = "drop_pixel_coords_sample18.csv"

# ==========================================
# FUNCTIONS
# ==========================================
def compute_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return 0.0
    return (4 * math.pi * area) / (perimeter * perimeter)

def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def create_median_background(file_list, num_samples):
    """Samples frames across the sequence to create a clean background."""
    print(f"Building background model from {num_samples} samples...")
    
    # Pick frames evenly distributed across the sequence
    indices = np.linspace(0, len(file_list) - 1, num_samples, dtype=int)
    sample_frames = []

    for idx in indices:
        img = cv2.imread(file_list[idx], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            sample_frames.append(img)
    
    # Calculate median across the stack
    median_bg = np.median(sample_frames, axis=0).astype(dtype=np.uint8)
    return median_bg

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    files = sorted(glob.glob(os.path.join(frames_dir, frame_pattern)))
    assert len(files) > 0, "No frames found."

    # 1. Generate the Master Background
    # This removes all moving drops and average out random sensor noise
    master_bg = create_median_background(files, num_bg_samples)
    master_bg = cv2.GaussianBlur(master_bg, blur_size, 0)
    
    # Save a copy for you to check
    cv2.imwrite("debug_master_background_sample18.png", master_bg)
    print("Master background created and saved as 'debug_master_background.png'")

    # 2. Tracking Setup
    track_state = "UNINITIALIZED"
    missed = 0
    last_pos = None
    last_velocity = (0.0, 0.0)
    output_rows = []

    # 3. Process Frames
    for frame_idx, filename in enumerate(files):
        img = cv2.imread(filename)
        if img is None: continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, blur_size, 0)
        time_s = frame_idx / fps

        # --- SUBTRACTION ---
        # Subtract the static master background from current frame
        diff = cv2.absdiff(gray, master_bg)
        
        # --- DETECTION ---
        _, bw = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Morphological opening to remove isolated noise pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area): continue
            
            circ = compute_circularity(cnt)
            if circ < circularity_min: continue

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
            
            confidence = circ * (area / max_area)
            candidates.append({"cx": cx, "cy": cy, "area": area, "circ": circ, "confidence": confidence})

        # --- TRACKING ---
        if track_state == "UNINITIALIZED":
            if candidates:
                chosen = max(candidates, key=lambda c: c["area"])
                track_state = "TRACKING"
                missed = 0
                last_pos = (chosen["cx"], chosen["cy"])
                last_velocity = (0.0, 0.0)
                status = "ok_init"
            else:
                chosen = None
                status = "no_detect"
        else:
            predicted = (last_pos[0] + last_velocity[0], last_pos[1] + last_velocity[1])
            valid = [c for c in candidates if get_distance((c["cx"], c["cy"]), predicted) <= gate_radius_px]
            
            if valid:
                chosen = min(valid, key=lambda c: get_distance((c["cx"], c["cy"]), predicted))
                last_velocity = (chosen["cx"] - last_pos[0], chosen["cy"] - last_pos[1])
                last_pos = (chosen["cx"], chosen["cy"])
                missed = 0
                status = "ok_track"
            else:
                missed += 1
                last_pos = predicted
                chosen = None
                status = "lost"
                if missed > max_missed_frames:
                    track_state = "UNINITIALIZED"

        # Save result
        if chosen:
            output_rows.append([frame_idx, os.path.basename(filename), time_s, 
                                chosen["cx"], chosen["cy"], chosen["area"], 
                                chosen["circ"], chosen["confidence"], status])
        else:
            output_rows.append([frame_idx, os.path.basename(filename), time_s, 
                                None, None, None, None, 0.0, status])

    # 4. Save CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "filename", "time_s", "x_px", "y_px", "area_px2", "circularity", "confidence", "status"])
        writer.writerows(output_rows)

if __name__ == "__main__":
    main()