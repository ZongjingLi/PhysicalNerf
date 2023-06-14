import torch
import torch.nn as nn

import numpy as np

class GridDynamics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.voxel_size = config.voxel_size
        self.points_per_voxel = config.points_per_voxel

    def state_evolve(self, state):
        return state
        