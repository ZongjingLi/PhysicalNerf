import torch
import torch.nn as nn

class VanillaNerf(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x, c2w):return x

