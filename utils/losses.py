import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-2):

        # comment out if the model contains a sigmoid  layer
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # for binary classification here
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice  # 1-score to have loss


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, preds, targets, smooth=1e-2):

        # comment out if the model contains a sigmoid layer
        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (preds.sum() + targets.sum() + smooth)

        # compute BCE loss
        BCE = F.binary_cross_entropy(preds, targets, reduction='mean')

        return BCE + dice_loss
