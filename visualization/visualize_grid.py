# show some 3d-point cloud data.
from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import torch.nn as nn


def visualize_points(x_points, file_name = "outputs/points.png"):
    fig = plt.figure()
    ax = Axes3D(fig)
    # create the 3d canvas and setup the normalize coordinate
    rang = 1.0
    ax.set_zlim(-rang,rang)
    ax.set_xlim(-rang,rang)
    ax.set_ylim(-rang,rang)

    # scatter points on the axis
    ax.scatter(x_points[:,0], x_points[:,1], x_points[:,2], color = "cyan")
    plt.savefig(file_name)

if __name__ == "__main__":
    points = torch.randn([100, 3]) * 0.2

    visualize_points(points)