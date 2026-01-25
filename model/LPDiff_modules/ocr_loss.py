import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add OCR project to path to import models
START_DIR = r"d:\ICPR2026\OCR"
if START_DIR not in sys.path:
    sys.path.insert(0, START_DIR)

from src.models import ResTranOCR
from configs import config as ocr_config_module

class OCRPerceptualLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ocr_loss_weight = opt['ocr_opt']['ocr_loss_weight']
        self.ckpt_path = opt['ocr_opt']['ocr_ckpt_path']
        
        print(f'Loading ResTranOCR checkpoint from: {self.ckpt_path}')
        
        # Load checkpoint to get config if possible, or use default with overrides
        checkpoint = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        
        # Determine config params (assuming standard ones for now or inferring from checkpoint)
        # Ideally we load the config used training.
        # However, ResTranOCR initialization requires specific args.
        # Let's try to infer or use defaults.
        # The report said: num_frames=5, img_height=32, img_width=128
        
        # We need to construct the model struct first to load weights.
        # Let's inspect checkpoint['config'] if it exists (trainer.py saves it!)
        config_dict = checkpoint.get('config', {})
        
        # Default fallback
        num_classes = 37 # 36 chars + 1 blank (default)
        if 'chars' in config_dict:
             num_classes = len(config_dict['chars']) + 1
             
        self.model = ResTranOCR(
            num_classes=num_classes,
            num_frames=config_dict.get('num_frames', 5),
            transformer_heads=config_dict.get('transformer_heads', 8),
            transformer_layers=config_dict.get('transformer_layers', 3),
            transformer_ff_dim=config_dict.get('transformer_ff_dim', 2048),
            dropout=config_dict.get('transformer_dropout', 0.1),
            use_stn=config_dict.get('use_stn', True),
            ctc_mid_channels=config_dict.get('ctc_head', {}).get('mid_channels', 1024) if isinstance(config_dict.get('ctc_head'), dict) else 1024,
            ctc_return_feats=False
        )

        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if needed
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.l1_loss = nn.L1Loss()
        self.num_frames = self.model.num_frames

    def forward(self, sr, hr):
        """
        sr, hr: [B, C, H, W] - Single frame images (RGB)
        We act as if these are 1-frame videos, or replicate them to match training.
        ResTranOCR expects [B, F, C, H, W].
        
        We will extract FEATURES from the Backbone (ResNet).
        Features shape: [B*F, 512, 1, W']
        """
        # Resize to expected input size (32, 128)
        # ResTranOCR default is 32x128.
        h_target, w_target = 32, 128
        
        # Normalize if necessary. LP-Diff output is likely [0, 1] or [-1, 1].
        # Verify normalization. OCR transforms use mean(0.5), std(0.5) -> scaled to [-1, 1].
        # LP-Diff typically works in [-1, 1] or [0, 1].
        # Assuming SR/HR are tensor images. If they are in [-1, 1], we are good.
        
        sr_resized = F.interpolate(sr, size=(h_target, w_target), mode='bilinear', align_corners=False)
        hr_resized = F.interpolate(hr, size=(h_target, w_target), mode='bilinear', align_corners=False)
        
        # To get robust features, we can use the STN+Backbone part.
        # Since we have single frames, we can just feed them as [B, 1, C, H, W]
        # BUT ResTranOCR requires exactly `num_frames` (5) for the assertion in forward.
        # We can bypass forward and call components directly to avoid replicating 5 times which is wasteful.
        
        # Helper to extract features
        def extract_feats(img):
            # img: [B, C, H, W]
            
            # 1. STN (if used)
            if self.model.use_stn:
                # STN expects [B, C, H, W] directly (it handles flattening logic internally if we passed sequences)
                # But self.model.stn expects input feature map.
                # In ResTranOCR.forward:
                # x_flat = x.view(b * f, c, h, w)
                # theta = self.stn(x_flat)
                
                theta = self.model.stn(img)
                grid = F.affine_grid(theta, img.size(), align_corners=False)
                img_aligned = F.grid_sample(img, grid, align_corners=False)
            else:
                img_aligned = img
                
            # 2. Backbone
            feats = self.model.backbone(img_aligned) # [B, 512, 1, W']
            return feats

        sr_feats = extract_feats(sr_resized)
        with torch.no_grad():
            hr_feats = extract_feats(hr_resized)
            
        loss = self.l1_loss(sr_feats, hr_feats) * self.ocr_loss_weight
        return loss
