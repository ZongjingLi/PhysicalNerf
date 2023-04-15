from config import *
from models import *

class UnknownArgumentError(Exception):
    def __init__(self):super().__init__()

def train(model, config, args):
    pass

argparser = argparse.ArgumentParser()
argparser.add_argument("--device",                      default = device)

argparser.add_argument("--model_name",                  default = "vanilla")
argparser.add_argument("--epoch",                       default = 1000)

args = argparser.parse_args()

if args.model_name == "vanilla":
    model = VanillaNERF(config)
else:
    raise UnknownArgumentError

train(model, config, args)