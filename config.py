import torch
import argparse 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",                         default = device)
parser.add_argument("--resolution",                     default = (128,128))
config = parser.parse_args(args = [])