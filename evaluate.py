from config import *
from models import *
from visualization import *


dynamic_grid = DynamicGrid(config)

mpm3d = MaterialPointModel3d()
mpm3d.reset()
mpm3d.gravity[None] = [0, 0, -1e-6] 

N = 9000; T = 10000
init_state = torch.randn([N,3]) * 0.001

curr_state = init_state
curr_color = torch.randn([curr_state.shape[0]]) ** 2

fig = plt.figure()
ax = Axes3D(fig)
l = 1

for t in range(T):
    for s in range(int(2e-3 // mpm3d.dt)):
            mpm3d.substep()

    new_state = mpm3d.x.to_numpy()
    curr_state = new_state
    ax.cla()
    ax.set_xlim(-l,l);ax.set_ylim(-l,l);ax.set_zlim(-l,l)

    color = "red"

    ax.scatter(curr_state[:,0], curr_state[:,1], curr_state[:,2], color = color)
    plt.pause(0.01)


