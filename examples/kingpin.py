import numpy as np
import matplotlib.pyplot as plt
from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch

robot = Kuka()
robot.tool = torch

N_PATH = 10
WIDTH = 0.1
TICKNESS = 0.01
LENGHT = 0.5
HEIGHT = 0.05  # obstacle height
XPOS, YPOS, ZPOS = 0.8, 0.0, 0.1

# ====================================
# PATH
# ====================================
from acrobotics.path import AxisAnglePt, TolerancedNumber
from acrobotics.path import FreeOrientationPt, TolEulerPt

path = []
pos_start = np.array([XPOS + WIDTH / 2 - TICKNESS, YPOS - LENGHT / 2, ZPOS + TICKNESS])
pos_stop = pos_start + np.array([0, LENGHT, 0])

# add offset along z_axis (weld wire stickout)
offset = np.array([-0.02 / np.sqrt(2), 0, 0.02 / np.sqrt(2)])
pos_start += offset
pos_stop += offset

rz_tol = TolerancedNumber(-np.pi / 2, np.pi / 2, samples=30)
z_axis = np.array([1, 0, -1])
z_axis = z_axis / np.linalg.norm(z_axis)

ry_tol = TolerancedNumber(np.pi / 2, np.pi, nominal=np.pi * 3 / 4, samples=20)

for s in np.linspace(0, 1, N_PATH):
    pos = (1 - s) * pos_start + s * pos_stop
    # path.append(AxisAnglePt(pos, z_axis , rz_tol, 0.0))
    # path.append(FreeOrientationPt(pos))
    path.append(TolEulerPt(pos, [0.0, ry_tol, rz_tol]))

# ====================================
# SCENE
# ====================================
from acrobotics.geometry import Shape, Scene

table = Shape(0.5, 0.5, 0.1)
l_bottom = Shape(WIDTH, LENGHT, TICKNESS)
l_top = Shape(TICKNESS, LENGHT, WIDTH)
obstacle = Shape(WIDTH / 2, LENGHT / 5, HEIGHT)

tf0 = np.array(
    [[1, 0, 0, XPOS - 0.20], [0, 1, 0, YPOS], [0, 0, 1, ZPOS - 0.05], [0, 0, 0, 1]]
)
tf1 = np.array(
    [[1, 0, 0, XPOS], [0, 1, 0, YPOS], [0, 0, 1, ZPOS + TICKNESS / 2], [0, 0, 0, 1]]
)

tf2 = np.array(
    [
        [1, 0, 0, XPOS + WIDTH / 2 - TICKNESS / 2],
        [0, 1, 0, YPOS],
        [0, 0, 1, ZPOS + WIDTH / 2 + TICKNESS],
        [0, 0, 0, 1],
    ]
)


obstacle_tf = np.array(
    [
        [1, 0, 0, XPOS - WIDTH / 4],
        [0, 1, 0, YPOS],
        [0, 0, 1, ZPOS + TICKNESS + HEIGHT / 2],
        [0, 0, 0, 1],
    ]
)

scene = Scene([table, l_bottom, l_top, obstacle], [tf0, tf1, tf2, obstacle_tf])


# ====================================
# SOLVE
# ====================================
from acrobotics.planning import cart_to_joint_no_redundancy
from acrobotics.planning import get_shortest_path

Q = cart_to_joint_no_redundancy(robot, path, scene, num_samples=400)

print([len(qi) for qi in Q])
qp = [qi[0] for qi in Q]

res = get_shortest_path(Q, method="dijkstra")
print(res)
qp_sol = res["path"]

# ====================================
# VISUALS
# ====================================
from acrobotics.util import get_default_axes3d, plot_reference_frame

# fig, ax = get_default_axes3d(xlim=[0, 1.5], ylim=[-0.75, 0.75], zlim=[0, 1.5])
# robot.plot(ax, [0, 1.5, 0, 0, 0, 0], c='k')
# scene.plot(ax, c='g')
# for pi in path: pi.plot(ax)
# plt.show(block=True)

fig2, ax2 = get_default_axes3d(xlim=[0, 1.5], ylim=[-0.75, 0.75], zlim=[0, 1.5])
for pi in path:
    pi.plot(ax2)
scene.plot(ax2, c="g")
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
