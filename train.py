from config import *
from models import *
from datasets import *
#from visualize import *

class UnknownArgumentError(Exception):
    def __init__(self):super().__init__()

def train(model, config, args):
    if args.dataset_name == "pac_static":
        dataset = PACStaticDataset(args.scene_name, resolution = config.resolution)
    if args.dataset_name == "pac_dynamic":
        dataset = PACDynamicDataset(args.scene_name, resolution = config.resolution)

    print("Experiment Started:")
    print("Static Frame Training")
    for epoch in range(args.epoch):
        frames = []
        for i in range(len(dataset)):
            sample = dataset[i]
            images = sample["images"]
            c2w    = sample["c2w"]

            outputs = model(images, c2w)

            #print(images.shape, c2w.shape)
            ##visualize_image_grid(images[:10],row = 5, save_name="nerf_views_{}".format(i))
            frames.append(Image.open("outputs/nerf_views_{}.png".format(i)))
        
        frame_one = frames[0]
        frame_one.save("outputs/nerf_views.gif", format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)
    torch.save(model, "checkpoints/{}_{}.ckpt".format(
        "dynamic" if int(args.dynamic) else "static",args.scene_name
        ))
    print("Training Ended.")

argparser = argparse.ArgumentParser()
argparser.add_argument("--device",                      default = device)

argparser.add_argument("--model_name",                  default = "vanilla")
argparser.add_argument("--dataset_name",                default = "pac_dynamic")
argparser.add_argument("--scene_name",                  default = "droplet")
argparser.add_argument("--dynamic",                     default = 1)
argparser.add_argument("--epoch",                       default = 1)

args = argparser.parse_args()

if args.model_name == "vanilla":
    model = VanillaNERF(config)
else:
    raise UnknownArgumentError

train(model, config, args)