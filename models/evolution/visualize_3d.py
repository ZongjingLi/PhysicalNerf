# show some 3d-point cloud data.
from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import torch.nn as nn

fig = plt.figure()
ax = Axes3D(fig)

N = 100
x_points = torch.randn([N, 3]) * 0.5 + torch.tensor([-0.,-0,5]).unsqueeze(0).repeat([N,1])
y_points = torch.randn([N, 3]) * 0.3 + torch.tensor([0.2,2,2]).unsqueeze(0).repeat([N,1])

scale = 1
x = torch.linspace(-1, 1, 100) * scale
y = torch.linspace(-1, 1, 100) * scale
X,Y = torch.meshgrid(x,y)
Z = X**2 + Y**2 

#ax.plot_surface(X,Y,Z, cmap = "rainbow")

ax.plot_surface(X,Y,Z * 0, cmap = "winter")


# initialize positions
a_pos = torch.tensor([-1.0, -1.0, 0.3])
b_pos = torch.tensor([1.0,  -1.0, 0.3]) 
# initualize velocities
a_velocity = torch.tensor([0.02,  0.02, 0.0]) * 0.05
b_velocity = torch.tensor([-0.02, 0.02, 0.0]) * 0.0

size = 0.01
eps = 0.01
dt = 1

def U(x,y):return -torch.mean(1 / (eps + torch.abs(x - y)**2 ))

time_steps = 1000

for i in range(time_steps):
    plt.cla()
    rang = 2.0
    # set up the limit of x,y,z and normalize them
    ax.set_zlim(0.0,1.0)
    ax.set_xlim(-rang,rang)
    ax.set_ylim(-rang,rang)

    # construct sample points of point clouds
    x_points = torch.randn([N, 3]) * size + (a_pos).unsqueeze(0).repeat([N,1])
    y_points = torch.randn([N, 3]) * size + (b_pos).unsqueeze(0).repeat([N,1])
    
    # evolve the point cloud
    coupling_constant = 0.0000065
    # update velocities
    force_to_a = -U(a_pos,b_pos) * b_pos

    new_a_velocity = a_velocity + coupling_constant * force_to_a
    a_velocity = new_a_velocity

    force_to_b = -U(a_pos,b_pos) * a_pos

    new_b_velocity = b_velocity + coupling_constant * force_to_b
    b_velocity = new_b_velocity

    # update positions
    a_pos = a_pos + a_velocity * dt
    b_pos = b_pos + b_velocity * dt

    # plot the ground surface
    ax.plot_surface(X,Y,Z *0, color = "grey", alpha = 0.1)

    # scatter plot the points after evolution
    ax.scatter(x_points[:,0], x_points[:,1], x_points[:,2], color = "red")
    ax.scatter(y_points[:,0], y_points[:,1], y_points[:,2], color = "cyan")
    plt.pause(0.01)
#ax.scatter(y_points[:,0], y_points[:,1], y_points[:,2], color = "cyan")

plt.show()
