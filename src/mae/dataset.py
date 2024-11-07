from PIL import Image

import glob

import torch
import torchvision.transforms as transforms

from transformers import AutoImageProcessor

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, is_training):
        self.args = args
        
        self.is_training = is_training
        
        image_processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast = True)

        if is_training:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale = (0.2, 1.0), interpolation = 3), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean = image_processor.image_mean, std = image_processor.image_std)])
        else:
            self.transform = image_processor

        if split == "train":
            self.images = sorted(glob.glob(f"data/{split}/*/*"))
        else:
            self.images = sorted(glob.glob(f"data/{split}/*"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        
        image = Image.open(image)
        image = image.convert("RGB")
        
        if self.is_training:
            image = self.transform(image)
        else:
            image = self.transform(image, return_tensors = "pt")["pixel_values"].squeeze(0)
        return image