import torch
import torch.nn as nn

class PhysicSimulation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.model_name = config.physics_model
        self.dt = config.dt

    def forward(self, state):
        """
        evolution of the point cloud states, input [N,3]
        """
        if self.model_name == "trivial":return state + torch.randn(state.shape) * self.dt
        
        if self.model_name == "cauchy":
            return state + 1/(torch.randn(state.shape) + 1) * self.dt