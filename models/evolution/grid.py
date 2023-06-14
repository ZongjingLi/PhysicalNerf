import torch
import torch.nn as nn

import numpy as np

class DynamicGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        voxel_size = config.voxel_size
        W, H, D = voxel_size
        self.voxel_size = config.voxel_size
        self.points_per_voxel = config.points_per_voxel

        x, y, z = torch.linspace(0, 1, W),\
            torch.linspace(0, 1, H),torch.linspace(0, 1, D)
        x, y, z = torch.meshgrid([x,y,z])
        print(x.shape)
        self.x = x

    def state_evolve(self, state):
        return state

 
