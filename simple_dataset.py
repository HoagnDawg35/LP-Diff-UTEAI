import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class LabelConverter:
    def __init__(self, alphabet):
        self.alphabet = alphabet  # "0123...abc..."
        self.dict = {}
        for i, char in enumerate(alphabet):
            # 0 is reserved for blank (required by CTCLoss)
            self.dict[char] = i + 1
            
    def encode(self, text):
        length = []
        result = []
        for t in text:
            length.append(len(t))
            for char in t:
                if char in self.dict:
                    index = self.dict[char]
                    result.append(index)
                else:
                    # Handle unknown chars? Ignore or map to unknown?
                    # For simplicty, ignore
                    pass
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class SimpleOCRDataset(Dataset):
    def __init__(self, root, transform=None, alphabet="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", mode='train'):
        self.root = root
        self.transform = transform
        self.alphabet = alphabet
        self.converter = LabelConverter(alphabet)
        self.mode = mode
        
        # Determine structure
        # Expecting: root/HR/*.png, root/LR/*.png 
        # or just single images if simplistic
        
        self.image_files = sorted(glob.glob(os.path.join(root, "*.jpg")) + glob.glob(os.path.join(root, "*.png")))
        
        # Try to find labels
        self.labels = {}
        label_file = os.path.join(root, "labels.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        name = parts[0]
                        txt = " ".join(parts[1:])
                        self.labels[name] = txt
        else:
            # Fallback: filename is label
            pass

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        fname = os.path.basename(path)
        
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = self.labels.get(fname, fname.split('.')[0]) # Default to filename
        
        # Encode label
        text, length = self.converter.encode([label])
        
        return img, text.flatten(), length.flatten()
        
    def collate_fn(self, batch):
        imgs, texts, lengths = zip(*batch)
        imgs = torch.stack(imgs, 0)
        texts = torch.cat(texts, 0)
        lengths = torch.cat(lengths, 0)
        return imgs, texts, lengths
