import torch
import torch.nn as nn

class VanillaNERF(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):return x