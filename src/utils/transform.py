import numpy as np

import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_fn(args, input, target):
    # generate mixed sample
    lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, (target_a, target_b), lam

def mixup_fn(args, input, target):
    # generate mixed sample
    lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()
    target_a = target
    target_b = target[rand_index]
    input = lam * input + (1 - lam) * input[rand_index, :]
    return input, (target_a, target_b), lam