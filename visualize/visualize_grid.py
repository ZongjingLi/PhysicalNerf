import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from matplotlib.patches import Rectangle

def visualize_image_grid(images, row, save_name = "image_grid"):
    plt.figure(save_name, frameon = False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    images = images.permute([0,3,1,2])
    comps_grid = torchvision.utils.make_grid(images,normalize=True,nrow=row).permute([1,2,0])

    plt.imshow(comps_grid.cpu().detach().numpy())
    plt.savefig("outputs/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)