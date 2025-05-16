import torch
import torch.nn as nn
import torch.nn.functional as F

class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        c = self.w - self.w * torch.log(1 + self.w / self.epsilon)
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1 + abs_x / self.epsilon),
            abs_x - c
        )
        return loss.mean()

def get_loss():
    return WingLoss() 