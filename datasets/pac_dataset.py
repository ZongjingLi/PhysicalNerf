import torch
from torch.utils.data import Dataset, DataLoader

class PACStaticDataset(Dataset):
    def __init__(self, resolution = (128,128)):
        super().__init__()
        self.resolution = resolution


    def __getitem__(self, idx):
        return idx


    def __len__(self): return 9