from config import *
from models import *
from visualization import *

dynamic_grid = DynamicGrid(config)

N = 1000; T = 10000
init_state = torch.randn([N,3]) * 0.001

curr_state = init_state

fig = plt.figure()
ax = Axes3D(fig)


for t in range(T):
    new_state = dynamic_grid.physics_model(curr_state)
    curr_state = new_state
    ax.cla()
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1)
    ax.scatter(curr_state[:,0], curr_state[:,1], curr_state[:,2])
    plt.pause(0.01)
