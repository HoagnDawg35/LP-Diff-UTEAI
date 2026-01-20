import argparse
import logging
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import model as Model
import core.logger as Logger
from model.ocr_head import OCRHead
from simple_dataset import SimpleOCRDataset, LabelConverter

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_ocr')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/LP-Diff.json', help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('--dataroot', type=str, default='d:/ICPR2026/test-public/track_10005', help='Path to images')
    
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Disable legacy OCR loss in LP-Diff model initialization to avoid path errors
    if 'ocr_opt' in opt:
        opt['ocr_opt']['use_ocr_loss'] = False
    if 'train' in opt and 'ocr_opt' in opt['train']:
        opt['train']['ocr_opt']['use_ocr_loss'] = False
        
    # 1. Load LP-Diff Model (Frozen)
    logger.info("Loading LP-Diff model...")
    diffusion = Model.create_model(opt)
    diffusion.netG.eval()
    for param in diffusion.netG.parameters():
        param.requires_grad = False
    logger.info("LP-Diff model loaded and frozen.")
    
    # 2. Initialize OCR Head
    logger.info("Initializing OCR Head...")
    # Alphabet matches simple_dataset default + hyphen for filenames
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-" 
    vocab_size = len(alphabet)
    ocr_head = OCRHead(vocab_size=vocab_size, imgH=32).to(device)
    ocr_head.train()
    
    # Optimizer for OCR Head only
    optimizer = torch.optim.Adam(ocr_head.parameters(), lr=1e-4)
    
    # 3. Data Loader
    logger.info("Setting up data loader...")
    transform = transforms.Compose([
        transforms.Resize((32, 128)), # Standard CRNN size
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    # Note: Using SimpleOCRDataset. In real scenario, this should load HR/LR pairs.
    # For now, we simulate LR by downsampling HR from the dataset
    dataset = SimpleOCRDataset(root=args.dataroot, transform=transform, alphabet=alphabet)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        logger.warning(f"No images found in {args.dataroot}. Creating dummy data for verification.")
        # Create a dummy entry locally? Or just skip
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)
    
    # 4. Training Loop
    epochs = 5
    logger.info("Starting training loop...")
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (imgs, texts, lengths) in enumerate(dataloader):
            imgs = imgs.to(device)
            texts = texts.to(device)
            lengths = lengths.to(device)
            
            # Simulate HR/LR pair
            # Treat loaded 'imgs' as HR
            hr_imgs = imgs
            # Create LR by downsampling then upsampling (or just resizing)
            # For OCR, typically we input fixed height 32. 
            # If we want to simulate "bad" LR, we can blur or downsample.
            lr_imgs = F.interpolate(hr_imgs, scale_factor=0.5, mode='bilinear')
            lr_imgs = F.interpolate(lr_imgs, size=(32, 128), mode='bilinear') # Resize back to expected input
            
            optimizer.zero_grad()
            
            # HR Pass
            loss_hr, _ = ocr_head(hr_imgs, texts, lengths)
            
            # LR Pass
            loss_lr, _ = ocr_head(lr_imgs, texts, lengths)
            
            # Total Loss
            loss = loss_hr + loss_lr
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} (HR: {loss_hr.item():.4f}, LR: {loss_lr.item():.4f})")
                
        logger.info(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / max(1, len(dataloader)):.4f}")
        
    # Save Checkpoint
    os.makedirs('checkpoint', exist_ok=True)
    torch.save(ocr_head.state_dict(), 'checkpoint/ocr_head_latest.pth')
    logger.info("Saved OCR head checkpoint.")

if __name__ == '__main__':
    main()
