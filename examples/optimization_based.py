import time
import numpy as np
from acrobotics.util import get_default_axes3d
from acrobotics.resources.path_on_table import scene2, scene1, path_ori_free
from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.optimization import get_optimal_path

robot = Kuka()
# robot.tool = torch
scene = scene1

N = 15  # path discretization
# TASK: the end-effector path (no orientation)
xp = np.ones(N) * 0.8
yp = np.linspace(-0.2, 0.2, N)
zp = np.ones(N) * 0.2

# Get initial values from task and inverse kinematics
Tee = np.eye(4)
Tee[:3, 3] = np.array([xp[0], yp[0], zp[0]])
q_ik = robot.ik(Tee)["sol"]

q_inits = [np.tile(qi, (N, 1)) for qi in q_ik]


qp_sol = get_optimal_path(path_ori_free, robot, scene, q_init=q_inits[5])["path"]

import matplotlib.pyplot as plt

fig, ax = get_default_axes3d([0, 1], [-0.5, 0.5], [0, 1])
ax.scatter(xp, yp, zp)
scene.plot(ax, c="g")
robot.animate_path(fig, ax, qp_sol)
ax.set_axis_off()
ax.view_init(24, 50)
plt.show()
