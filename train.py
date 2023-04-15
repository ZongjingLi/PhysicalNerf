from config import *

def train(model, config, args):
    pass

argparser = argparse.ArgumentParser()
argparser.add_argument("--device",                      default = device)
argparser.add_argument("--epoch",                       default = 1000)

args = argparser.parse_args()


model = None

train(model, config, args)