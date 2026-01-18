import os
import glob
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_icpr_to_lp_diff():
    # Define paths
    source_root = r"h:\ICPR2026\ICPR_dataset\train"
    # User requested nested structure: icpr_train/train and icpr_train/val
    base_target_root = r"h:\ICPR2026\LP-Diff-UTEAI\data\icpr_train"
    
    # Define output directories
    target_train_root = os.path.join(base_target_root, "train")
    target_val_root = os.path.join(base_target_root, "val")
    
    print(f"Scanning for tracks in {source_root}...")
    
    # Recursively find all 'track_XXXXX' folders
    track_dirs = []
    for root, dirs, files in os.walk(source_root):
        for d in dirs:
            if d.startswith("track_"):
                track_dirs.append(os.path.join(root, d))
    
    track_dirs.sort()
    print(f"Found {len(track_dirs)} tracks.")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(track_dirs)
    
    val_size = 5000
    if len(track_dirs) < val_size:
        print("Warning: Dataset smaller than requested validation size. Using 20% for validation.")
        val_size = int(len(track_dirs) * 0.2)
        
    val_tracks = track_dirs[:val_size]
    train_tracks = track_dirs[val_size:]
    
    print(f"Splitting into {len(train_tracks)} training and {len(val_tracks)} validation tracks.")
    
    # Helper function to process a list of tracks
    def process_tracks(tracks, root_dir, phase_name):
        inputs_dir = os.path.join(root_dir, "inputs")
        gt_dir = os.path.join(root_dir, "gt")
        
        os.makedirs(inputs_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        
        plate_counter = 0
        
        for track_path in tqdm(tracks, desc=f"Converting {phase_name}"):
            try:
                # Get LR and HR images
                files = os.listdir(track_path)
                lr_files = sorted([f for f in files if f.startswith("lr-") and f.endswith(".png")])
                hr_files = sorted([f for f in files if f.startswith("hr-") and f.endswith(".png")])
                
                if not lr_files:
                    continue
                if not hr_files:
                    continue
                
                # Create plate directory
                plate_id = f"plate_{plate_counter}"
                plate_input_dir = os.path.join(inputs_dir, plate_id)
                plate_gt_dir = os.path.join(gt_dir, plate_id)
                
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
                
                # Process GT (First HR image)
                # mid_idx = len(hr_files) // 2
                gt_file = hr_files[0]
                
                src_gt_path = os.path.join(track_path, gt_file)
                dst_gt_name = f"img_{input_count}.jpg" 
                dst_gt_path = os.path.join(plate_gt_dir, dst_gt_name)
                
                with Image.open(src_gt_path) as img:
                    img.convert('RGB').save(dst_gt_path, "JPEG", quality=95)
                
                plate_counter += 1
                
            except Exception as e:
                print(f"Error processing {track_path}: {e}")
                continue
                
        print(f"Processed {plate_counter} plates for {phase_name} in {root_dir}")

    # Run processing
    process_tracks(train_tracks, target_train_root, "TRAIN")
    process_tracks(val_tracks, target_val_root, "VAL")

    print(f"Full conversion complete. Data saved to {base_target_root}")

if __name__ == "__main__":
    convert_icpr_to_lp_diff()
