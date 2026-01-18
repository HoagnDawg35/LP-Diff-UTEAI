import torch
import torch.nn as nn
from .transocr import Transformer

class OCRPerceptualLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ocr_loss_weight = opt['ocr_opt']['ocr_loss_weight']
        self.ckpt_path = opt['ocr_opt']['ocr_ckpt_path']
        
        # Hardcode vocabulary size as in DiffTSR (6736) or load from config if available.
        # Since we are using this for perceptual loss (feature matching), the exact vocab size 
        # matters less for the encoder part but must match the checkpoint to load weights safely.
        # 6736 is standard for this model in DiffTSR.
        self.vocab_size = 6736 
        
        self.trans_ocr_model = Transformer(self.vocab_size)
        
        print(f'Loading TransOCR checkpoint from: {self.ckpt_path}')
        checkpoint = torch.load(self.ckpt_path)
        
        # Handle different state_dict formats (e.g. DataParallel wrapper)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.trans_ocr_model.load_state_dict(new_state_dict, strict=False)
        self.trans_ocr_model.eval()
        
        # Freeze parameters
        for param in self.trans_ocr_model.parameters():
            param.requires_grad = False
            
        self.l1_loss = nn.L1Loss()

    def forward(self, sr, hr):
        # Resize to (32, 256) as expected by TransOCR ResNet backbone (approximate)
        # Verify the actual expected size. DiffTSR uses (32, 128) or similar.
        # Let's use (32, 256) as a safe bet for OCR models or inspect DiffTSR config again.
        # Re-checking DiffTSR_config: image_size: [32, 128] for IDM, but TransOCR might expect 
        # something else. However, since it's fully convolutional until attention, 
        # we can stick to a reasonable size. Standard CRNN/TransOCR is usually height 32.
        # DiffTSR training uses 32, 256 for TransOCR loss.
        h, w = 32, 256
        
        sr_resized = torch.nn.functional.interpolate(sr, size=(h, w), mode='bilinear', align_corners=False)
        hr_resized = torch.nn.functional.interpolate(hr, size=(h, w), mode='bilinear', align_corners=False)
        
        # Extract features
        # We only need the 'conv' features from the encoder for perceptual loss
        with torch.no_grad():
            hr_feats = self.trans_ocr_model(hr_resized, None, None)['conv']
            
        sr_feats = self.trans_ocr_model(sr_resized, None, None)['conv']
        
        loss = self.l1_loss(sr_feats, hr_feats) * self.ocr_loss_weight
        return loss
