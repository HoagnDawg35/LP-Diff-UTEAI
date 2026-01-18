import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

import logging
logging.basicConfig(level=logging.INFO)

from model.LPDiff_modules.ocr_loss import OCRPerceptualLoss
from model.model import DDPM

def verify():
    print("Verifying OCR Perceptual Loss implementation...")
    
    # Mock options
    opt = {
        'name': 'Verify_OCR',
        'phase': 'train',
        'gpu_ids': [0],
        'distributed': False,
        'path': {
            'log': 'logs',
            'tb_logger': 'tb_logger',
            'results': 'results',
            'checkpoint': 'checkpoint',
            'resume_state': None
        },
        'datasets': {},
        'model': {
            'finetune_norm': False,
            'beta_schedule': {'train': {'schedule': 'linear', 'n_timestep': 1000, 'linear_start': 1e-4, 'linear_end': 2e-2}},
            'diffusion': {
                'image_size': 128,
                'channels': 3,
                'conditional': True
            },
            'unet': {
                'in_channel': 6,
                'out_channel': 3,
                'inner_channel': 32,
                'channel_multiplier': [1, 2, 4],
                'attn_res': [16],
                'res_blocks': 1,
                'dropout': 0.1,
                'norm_groups': 32
            }
        },
        'train': {
            'optimizer': {'lr': 1e-4},
            'use_prerain_MTA': False,
            'resume_training': False
        },
        'ocr_opt': {
            'use_ocr_loss': True,
            'ocr_loss_weight': 0.01,
            'ocr_ckpt_path': 'H:/ICPR2026/DiffTSR/ckpt/transocr.pth'
        }
    }
    
    try:
        # 1. Instantiate OCR Loss
        print("Instantiating OCRPerceptualLoss...")
        ocr_loss = OCRPerceptualLoss(opt)
        print("OCRPerceptualLoss instantiated successfully.")
        
        # 2. Instantiate DDPM Model (which includes GaussianDiffusion and initialized OCR loss)
        print("Instantiating DDPM Model...")
        model = DDPM(opt)
        print("DDPM Model instantiated successfully.")
        
        # 3. Test forward pass of OCR loss with dummy data
        print("Testing OCR Loss forward pass...")
        if torch.cuda.is_available():
            ocr_loss = ocr_loss.cuda()
            dummy_sr = torch.randn(1, 3, 128, 128).cuda()
            dummy_hr = torch.randn(1, 3, 128, 128).cuda()
        else:
            dummy_sr = torch.randn(1, 3, 128, 128)
            dummy_hr = torch.randn(1, 3, 128, 128)
            
        loss = ocr_loss(dummy_sr, dummy_hr)
        print(f"OCR Loss computed: {loss.item()}")
        
        print("Verification passed!")
        
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
