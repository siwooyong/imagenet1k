import torch
import torch.nn as nn

import timm

class CustomModel(nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.backbone = timm.create_model(
            model_name = args.model_name,
            pretrained = args.pretrained,
            drop_path_rate = args.drop_path_rate,
            )

    def forward(self, x):
        x = self.backbone(x)
        return x