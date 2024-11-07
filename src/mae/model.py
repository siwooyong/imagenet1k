import glob

import torch
import torch.nn as nn

from transformers import ViTMAEConfig, ViTMAEForPreTraining

class CustomModel(nn.Module):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        self.args = args
        
        if args.stage == "pretrain":
            config = ViTMAEConfig.from_pretrained(args.model_name)
            config.norm_pix_loss = args.norm_pix_loss
            
            self.backbone = ViTMAEForPreTraining(config)
            
        elif args.stage == "finetune":
            config = ViTMAEConfig.from_pretrained(args.model_name)
            config.mask_ratio = 0.0
            
            checkpoint = sorted(glob.glob("./" + args.pretrained_dir + "/*.bin"))[-1]
            checkpoint = torch.load(checkpoint, weights_only = True, map_location = "cpu")
            checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items() if k.startswith("backbone.")}
            
            pretrained_model = ViTMAEForPreTraining(config)
            pretrained_model.load_state_dict(checkpoint)
            
            self.backbone = pretrained_model.vit
            
            self.out = nn.Linear(config.hidden_size, args.n_class)
            
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.backbone(x)
        
        if self.args.stage == "finetune":
            x = x.last_hidden_state
            x = x[:, 1:]
            x = x.mean(1)
            x = self.out(x)
        return x