import pandas as pd

from PIL import Image

import os
import glob

import matplotlib.pyplot as plt

import torch

import timm

from src.utils.classes import IMAGENET2012_CLASSES

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, is_training):
        self.args = args

        self.split = split

        if is_training:
            self.transform = timm.data.create_transform(
                input_size = args.input_size,
                color_jitter = args.color_jitter,
                auto_augment = args.aa,
                interpolation = args.train_interpolation,
                re_prob = args.reprob,
                re_mode = args.remode,
                re_count = args.recount,
                is_training = is_training,
                )
        else:
            data_config = timm.data.resolve_model_data_config(args.model_name)
            self.transform = timm.data.create_transform(**data_config, is_training = is_training)

        self.df = self.get_df()

    def get_df(self):
        if self.split == "train":
            images = sorted(glob.glob(f"data/{self.split}/*/*"))
        else:
            images = sorted(glob.glob(f"data/{self.split}/*"))

        labels = []
        for image in images:
            root, _ = os.path.splitext(image)
            _, synset_id = os.path.basename(root).rsplit("_", 1)
            label = IMAGENET2012_CLASSES[synset_id]
            label = list(IMAGENET2012_CLASSES.values()).index(label)
            labels.append(label)

        df = pd.DataFrame()
        df["image"] = images
        df["label"] = labels
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image, label = self.df.loc[index]
        
        image = Image.open(image)
        image = image.convert("RGB")
        image = self.transform(image)

        label = torch.tensor(label, dtype = torch.long)
        return image, label