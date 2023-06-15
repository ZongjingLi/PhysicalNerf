import torch
import torch.nn as nn

import numpy as np
from .physics_simulation import *

class DynamicGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        voxel_size = config.voxel_size
        W, H, D = voxel_size

        # setup the voxel and sample config.
        self.voxel_size = config.voxel_size
        self.points_per_voxel = config.points_per_voxel

        # construct the normalize voxel grid configuration.
        x, y, z = torch.linspace(0, 1, W),\
            torch.linspace(0, 1, H),torch.linspace(0, 1, D)
        x, y, z = torch.meshgrid([x,y,z])
        self.x = x; self.y = y; self.z = z

    def point2grid(self, lagrange_state):
        return 0

    def grid2point(self, eulrer_state):
        return 0


    def state_evolve(self, state):
        return state

 
