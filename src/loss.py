import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, args, mode):
        super(CustomLoss, self).__init__()
        self.args = args

        if mode == "train":
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, preds, trues):
        loss = self.loss_fn(preds, trues)
        return loss