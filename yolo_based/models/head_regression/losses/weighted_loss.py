import torch
import torch.nn as nn

def get_loss(weight=None):
    if weight is not None:
        # User should apply weights in the training loop
        return nn.MSELoss(reduction='none')
    else:
        return nn.MSELoss() 