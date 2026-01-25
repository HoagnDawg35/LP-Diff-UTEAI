import os
import shutil
import random
import glob
from tqdm import tqdm

def split_data(train_root, val_root, split_ratio=0.25):
    """
    Moves `split_ratio` of track folders from train_root to val_root.
    Preserves directory structure (Scenario-X/Type/track_id).
    """
    print(f"Scanning {train_root}...")
    
    # 1. Find all tracks (recursively)
    # Structure: train_root/Scenario-X/Type/track_id
    # We want to identify "track_*" folders.
    track_pattern = os.path.join(train_root, "**", "track_*")
    all_tracks = glob.glob(track_pattern, recursive=True)
    
    # Filter only directories
    all_tracks = [t for t in all_tracks if os.path.isdir(t)]
    
    total_tracks = len(all_tracks)
    if total_tracks == 0:
        print("No tracks found!")
        return

    print(f"Found {total_tracks} tracks.")
    
    # 2. Shuffle and Select
    target_count = int(total_tracks * split_ratio)
    val_tracks = random.sample(all_tracks, target_count)
    
    print(f"Moving {len(val_tracks)} tracks to {val_root}...")
    
    # 3. Move
    for track_path in tqdm(val_tracks):
        # Calculate relative path to preserve structure
        # e.g. track_path = .../train/Scenario-A/Brazilian/track_001
        # rel_path = Scenario-A/Brazilian/track_001
        rel_path = os.path.relpath(track_path, train_root)
        dest_path = os.path.join(val_root, rel_path)
        
        # Create parent dirs
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Move
        try:
            shutil.move(track_path, dest_path)
        except Exception as e:
            print(f"Error moving {track_path}: {e}")

    print("Done!")

if __name__ == "__main__":
    TRAIN_ROOT = r"d:\ICPR2026\LP-Diff-UTEAI\data\train"
    VAL_ROOT = r"d:\ICPR2026\LP-Diff-UTEAI\data\val"
    
    # Ensure VAL_ROOT is empty or handle merge? 
    # For now assume fresh split.
    os.makedirs(VAL_ROOT, exist_ok=True)
    
    split_data(TRAIN_ROOT, VAL_ROOT, split_ratio=0.25)
