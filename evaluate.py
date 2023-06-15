from config import *
from models import *
from visualization import *

dynamic_grid = DynamicGrid(config)

N = 100; T = 100
init_state = torch.randn([N,3])

curr_state = init_state

fig = plt.figure()
ax = Axes3D(fig)

for t in range(T):
    new_state = dynamic_grid.physics_model(curr_state)
    curr_state = new_state
    ax.cla()
    ax.scatter()
    ax.pause(0.01)