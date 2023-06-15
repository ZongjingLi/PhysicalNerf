import torch
import argparse 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",                         default = device)
parser.add_argument("--resolution",                     default = (128,128))

# grid-evolution configuration
parser.add_argument("--voxel_size",                     default = (64,64,64))
parser.add_argument("--points_per_voxel",               default = 5)

# physics simulation
parser.add_argument("--physics_model",                  default = "trivial")
parser.add_argument("--dt",                             default = 0.01)
parser.add_argument("--init_shear",                     default = 1.0)
parser.add_argument("--init_bulk",                      default = 1.0)
config = parser.parse_args(args = [])