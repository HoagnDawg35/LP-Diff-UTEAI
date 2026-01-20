import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        # Use a lightweight ResNet feature extractor or VGG
        # Here we use a custom VGG-like extractor for simplicity and standard CRNN behavior
        # Or we can use ResNet18
        
        self.conv = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64xH/2xW/2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128xH/4xW/4
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 256xH/8xW/4 (Width stride 1)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)), # 512xH/16xW/4
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True) # 512xH/32xW/4
        )
        
        # After conv, if imgH=32, height is 1. Shape: [B, 512, 1, W_feat]
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # input: [B, C, H, W]
        conv = self.conv(input)
        
        b, c, h, w = conv.size()
        # print('Conv out:', b,c,h,w) 
        assert h == 1, "The height of conv must be 1"
        conv = conv.squeeze(2) # [B, C, W]
        conv = conv.permute(2, 0, 1)  # [W, B, C]
        
        output = self.rnn(conv) # [W, B, nClass]
        
        # Output log_softmax for CTC Loss
        # output = F.log_softmax(output, dim=2)
        
        return output

class OCRHead(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, imgH=32, input_channel=3):
        super(OCRHead, self).__init__()
        self.vocab_size = vocab_size
        self.model = CRNN(imgH, input_channel, vocab_size + 1, hidden_size) # +1 for blank
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True) # Assuming blank=0

    def forward(self, images, targets=None, target_lengths=None):
        # images: [B, C, H, W]
        # targets: [B, L] encoded labels
        # target_lengths: [B] actual lengths
        
        logits = self.model(images) # [T, B, C]
        
        if targets is not None:
            # Prepare args for CTC Loss
            T, B, _ = logits.size()
            input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long, device=images.device)
            log_probs = F.log_softmax(logits, dim=2)
            
            loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            return loss, logits
        else:
            return logits

    def predict(self, images, converter):
        # Helper to decode
        pass

