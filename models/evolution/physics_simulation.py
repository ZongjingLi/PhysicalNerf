import torch
import torch.nn as nn

class PhysicSimulation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        model_name = config.physics_model