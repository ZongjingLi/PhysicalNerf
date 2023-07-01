from config import *
from models import *
from visualization import *

# 英雄，那是你的过去.......
# 大地即将崩裂，天空也要为之哭泣！很快，艾泽拉斯那悲戚的血泪就会降下！
# 天灾军团的士兵们，阿彻鲁斯的死亡骑士们，黑暗的仆从们：听从大领主的召唤！起来吧！
# 英雄那是你的过去。你曾勇敢而无畏的对抗黑暗，
# 但你所对抗的黑暗并非轻易就能驱除，一切胜利果实也并非轻易就能守护。
# 如今死亡的阴影再次笼罩世界，邪恶的力量为了最终统治一切找到了新的仆从
# -以符文之力散播毁灭与死亡，忠实执行巫妖王命令的黑暗骑士们，
# 这是他们崛起的时刻这是你获得黑暗新生的时刻

dynamic_grid = DynamicGrid(config)

N = 1000; T = 10000
init_state = torch.randn([N,3]) * 0.001

curr_state = init_state
curr_color = torch.randn([curr_state.shape[0]]) ** 2

fig = plt.figure()
ax = Axes3D(fig)


for t in range(T):
    new_state = dynamic_grid.physics_model(curr_state)
    curr_state = new_state
    ax.cla()
    ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_zlim(-1,1)
    alphas = torch.randn([new_state.shape[0]]) ** 2
    colors = curr_color + torch.randn([new_state.shape[0]]) * 0.05
    alphas = alphas / alphas.max()
    colors = colors / colors.max()
    curr_color = colors
    ax.scatter(curr_state[:,0], curr_state[:,1], curr_state[:,2] ,c = colors, cmap = "rainbow")
    plt.pause(0.01)

