import os
import glob
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_icpr_to_lp_diff():
    # Define paths
    source_root = r"h:\ICPR2026\ICPR_dataset\train"
    target_root = r"h:\ICPR2026\LP-Diff-UTEAI\data\icpr_train"
    
    target_inputs = os.path.join(target_root, "inputs")
    target_gt = os.path.join(target_root, "gt")
    
    # Create target directories
    os.makedirs(target_inputs, exist_ok=True)
    os.makedirs(target_gt, exist_ok=True)
    
    print(f"Scanning for tracks in {source_root}...")
    
    # Recursively find all 'track_XXXXX' folders
    # Using glob to find directories starting with 'track_'
    track_dirs = []
    for root, dirs, files in os.walk(source_root):
        for d in dirs:
            if d.startswith("track_"):
                track_dirs.append(os.path.join(root, d))
    
    track_dirs.sort()
    print(f"Found {len(track_dirs)} tracks.")
    
    # Process each track
    plate_counter = 0
    
    for track_path in tqdm(track_dirs, desc="Converting tracks"):
        try:
            # Get LR and HR images
            files = os.listdir(track_path)
            lr_files = sorted([f for f in files if f.startswith("lr-") and f.endswith(".png")])
            hr_files = sorted([f for f in files if f.startswith("hr-") and f.endswith(".png")])
            
            if not lr_files:
                # print(f"Skipping {track_path}: No LR images found.")
                continue
                
            if not hr_files:
                # print(f"Skipping {track_path}: No HR images found.")
                continue
            
            # Create plate directory
            plate_id = f"plate_{plate_counter}"
            plate_input_dir = os.path.join(target_inputs, plate_id)
            plate_gt_dir = os.path.join(target_gt, plate_id)
            
            os.makedirs(plate_input_dir, exist_ok=True)
            os.makedirs(plate_gt_dir, exist_ok=True)
            
            # Process Inputs (LR images)
            input_count = 0
            for i, lr_file in enumerate(lr_files):
                src_img_path = os.path.join(track_path, lr_file)
                dst_img_name = f"img_{i}.jpg"
                dst_img_path = os.path.join(plate_input_dir, dst_img_name)
                
                with Image.open(src_img_path) as img:
                    img.convert('RGB').save(dst_img_path, "JPEG", quality=95)
                input_count += 1
            
            # Process GT (Middle HR image)
            # Find the middle index
            mid_idx = len(hr_files) // 2
            gt_file = hr_files[mid_idx]
            
            src_gt_path = os.path.join(track_path, gt_file)
            # GT filename follows the sequence (img_N, where N is count of inputs)
            dst_gt_name = f"img_{input_count}.jpg" 
            dst_gt_path = os.path.join(plate_gt_dir, dst_gt_name)
            
            with Image.open(src_gt_path) as img:
                img.convert('RGB').save(dst_gt_path, "JPEG", quality=95)
            
            plate_counter += 1
            
        except Exception as e:
            print(f"Error processing {track_path}: {e}")
            continue

    print(f"Conversion complete. Processed {plate_counter} plates.")
    print(f"Data saved to {target_root}")

if __name__ == "__main__":
    convert_icpr_to_lp_diff()
