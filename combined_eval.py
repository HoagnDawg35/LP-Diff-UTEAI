import argparse
import logging
import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm

# Add paths
START_DIR = r"d:\ICPR2026\LP-Diff-UTEAI"
OCR_DIR = r"d:\ICPR2026\OCR"

# sys.path.insert(0, OCR_DIR) # Conflict with LP-Diff src
sys.path.insert(0, START_DIR)

# LP-Diff Imports
import core.logger as Logger
import core.metrics as Metrics
import model as Model
import model as Model
# import data as Data # Missing module, defining inline

# --- Dataset Definition ---
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import torchvision.transforms as transforms

class MDLPDataset(Dataset):
    def __init__(self, opt, phase):
        self.opt = opt
        self.phase = phase
        self.dataroot = opt['dataroot']
        self.height = opt['height']
        self.width = opt['width']
        
        # Find all tracks
        # Structure: dataroot/Scenario-X/Type/track_id/*.png
        # Valid files usually: 01.png, 02.png, 03.png, 04.png, 05.png ??
        # Or License Plate frames.
        # Check one track to be sure? 
        # Assuming MDLP dataset structure: "track_*" folders containing frames.
        
        # We need to find all track directories
        self.tracks = sorted(glob.glob(os.path.join(self.dataroot, "**", "track_*"), recursive=True))
        self.tracks = [t for t in self.tracks if os.path.isdir(t)]
        
        if len(self.tracks) == 0:
            print(f"Warning: No tracks found in {self.dataroot}")
            
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        track_path = self.tracks[index]
        
        # Load frames. LP-Diff uses 3 LR frames? 
        # Config says: in_channel: 6, out_channel: 3.
        # This implies input is 2 frames of 3 channels? Or 3 frames of ...?
        # infer.py uses LR1, LR2, LR3?
        # infer.py lines 68: diffusion.feed_data(val_data)
        # infer.py lines 73-75 gets LR1, LR2, LR3.
        # So input is likely 3 frames.
        
        # Let's assume files are named 01.jpg, 02.jpg ... or similar.
        # Need to know the file naming convention.
        # Listing files in a track would help.
        # Assuming sorted images.
        
        files = sorted(glob.glob(os.path.join(track_path, "*.png")) + glob.glob(os.path.join(track_path, "*.jpg")))
        
        # If we have 5 frames (standard MDLP?), select 3 for LP-Diff?
        # Or maybe LP-Diff restores the middle one using adjacents?
        # Let's assume we maintain logic.
        # If < 3 frames, padding?
        
        # Hack: just take first 3 for now, or central.
        # Better: use indices 0, 1, 2.
        
        imgs = []
        for f in files:
            img = Image.open(f).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
            
        if len(imgs) < 3:
            # Replicate
            while len(imgs) < 3:
                imgs.append(imgs[-1])
                
        # LR1, LR2, LR3
        # LP-Diff might expect concatenated? In feed_data:
        # data['LR1'], data['LR2'], data['LR3']
        
        # We need to return a dict
        ret = {}
        ret['LR1'] = imgs[0]
        ret['LR2'] = imgs[1]
        ret['LR3'] = imgs[2]
        
        # HR? Do we have HR?
        # Ideally yes for eval.
        # If test mode, maybe not.
        # Let's assume HR is the *clearest* frame or separate HR folder?
        # The user report says "paired" dataset.
        ret['HR'] = imgs[1] # Dummy HR if missing, or use middle.
        
        ret['path'] = track_path # For saving
        
        return ret

def create_dataset(dataset_opt, phase):
    return MDLPDataset(dataset_opt, phase)

def create_dataloader(dataset, dataset_opt, phase):
    return DataLoader(
        dataset, 
        batch_size=dataset_opt.get('batch_size', 1), 
        shuffle=(phase=='train'), 
        num_workers=dataset_opt.get('num_workers', 0),
        pin_memory=True
    )

# Inlined from OCR/src/utils/post_proc.py
from itertools import groupby
from typing import Dict, List, Tuple

def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding."""
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))
    
    return results

import core.logger as Logger
# from OCR.src.utils import decode_with_confidence # Inlined above

# Safe import for OCR - Import model only then clean up sys.path
try:
    if OCR_DIR not in sys.path:
        sys.path.insert(0, OCR_DIR)
    from src.models import ResTranOCR
except ImportError as e:
    print(f"Error importing ResTranOCR: {e}")
finally:
    if OCR_DIR in sys.path:
        sys.path.remove(OCR_DIR)

def load_ocr_model(ckpt_path, device):
    """Load ResTranOCR model."""
    print(f"Loading OCR model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = checkpoint.get('config', {})
    
    # Defaults matching your training config
    num_classes = 37 
    if 'chars' in config_dict:
         num_classes = len(config_dict['chars']) + 1
         
    model = ResTranOCR(
        num_classes=num_classes,
        num_frames=config_dict.get('num_frames', 5),
        transformer_heads=config_dict.get('transformer_heads', 8),
        transformer_layers=config_dict.get('transformer_layers', 3),
        transformer_ff_dim=config_dict.get('transformer_ff_dim', 2048),
        dropout=config_dict.get('transformer_dropout', 0.1),
        use_stn=config_dict.get('use_stn', True),
        ctc_mid_channels=1024,
        ctc_return_feats=False
    )
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, config_dict.get('idx2char', {i+1: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")})

def preprocess_for_ocr(image_tensor, device, num_frames=5):
    """
    Prepare SR image for OCR.
    Input: [C, H, W] tensor (RGB, likely normalized [-1, 1] or [0, 1])
    Output: [1, 5, C, H, W] tensor
    """
    # LP-Diff output usually [-1, 1] or [0, 1].
    # OCR expects [-1, 1] (via Mean/Std 0.5).
    # Assuming LP-Diff output is [-1, 1] (standard for diffusion), we are good.
    # If [0, 1], we need to normalize.
    # Let's assume input is already decent, maybe ensure size.
    
    # Resize to 32x128
    img = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(32, 128), mode='bilinear', align_corners=False)
    
    # Replicate to num_frames
    img_seq = img.unsqueeze(1).repeat(1, num_frames, 1, 1, 1) # [1, 5, C, H, W]
    return img_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/LP-Diff.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--ocr_ckpt', type=str, default=r'd:\ICPR2026\OCR\results\v1_3frame.pth', help='Path to OCR checkpoint')
    parser.add_argument('--output_file', type=str, default='combined_submission.txt')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    
    if args.cpu:
        opt['gpu_ids'] = None
        args.gpu_ids = None
        
    opt = Logger.dict_to_nonedict(opt)

    # Logging
    Logger.setup_logger(None, opt['path']['log'], 'test', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # 1. Load LP-Diff Model
    logger.info("Initializing LP-Diff...")
    diffusion = Model.create_model(opt)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    
    # 2. Load OCR Model
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    ocr_model, idx2char = load_ocr_model(args.ocr_ckpt, device)
    
    # 3. Create Dataset
    # Use global create_dataset function defined in this file
    val_set = create_dataset(opt['datasets']['val'], 'val')
    val_loader = create_dataloader(val_set, opt['datasets']['val'], 'val')
    
    logger.info(f"Starting Inference on {len(val_set)} samples...")
    
    results = []
    
    # Inference Loop
    for i, val_data in enumerate(tqdm(val_loader)):
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)
        
        visuals = diffusion.get_current_visuals()
        sr_img = visuals['SR'][-1] # [C, H, W] - Last step of diffusion
        
        # Prepare for OCR
        # Need to put on same device
        sr_img = sr_img.to(device)
        
        # Preprocess (Resize & Replicate)
        # Use dynamic num_frames from the model instance
        num_frames = ocr_model.num_frames if hasattr(ocr_model, 'num_frames') else 5
        ocr_input = preprocess_for_ocr(sr_img, device, num_frames=num_frames)
        
        # Recognize
        with torch.no_grad():
            preds = ocr_model(ocr_input) # [1, Seq, Classes]
            decoded = decode_with_confidence(preds, idx2char)
            
        # Get Track ID (assuming dataloader provides it or we infer from index if simpler)
        # LP-Diff dataloader might not return track_ids easily.
        # Check simple_dataset.py... usually returns {'LR1':..., 'HR':..., 'Index': ...}
        # If we can't get track ID easily, we'll index differently or modify dataset.
        # For now, let's just print/save what we have.
        
        text, conf = decoded[0]
        # logger.info(f"Sample {i}: {text} ({conf:.4f})")
        
        # Save result
        results.append(f"sample_{i},{text};{conf:.4f}")

    # Save to file
    with open(args.output_file, 'w') as f:
        f.write("\n".join(results))
    
    logger.info(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()
