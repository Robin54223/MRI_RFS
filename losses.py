import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5
        input_flat = input  # torch.softmax(input, dim=1)
        target_flat = target

        intersection = input_flat * target_flat

        # w = 1 / (target_flat.sum((2,3,4))+smooth)
        # w = w / w.sum()
        loss = 2 * intersection.sum((2, 3, 4)) / (input_flat.sum((2, 3, 4)) + target_flat.sum((2, 3, 4)) + smooth)

        loss = torch.mean(1 - loss)

        return loss
