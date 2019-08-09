import time
import numpy as np
import casadi as ca
from casadi import cos, sin, dot
from acrobotics.util import get_default_axes3d
from acrobotics.resources.path_on_table import scene2, scene1
from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.optimization import create_cc

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


time_before = time.time()

opti = ca.Opti()
q = opti.variable(N, 6)  #  joint variables along path

# collision constraints
cons = create_cc(opti, robot, scene, q)
opti.subject_to(cons)

# create path constraints
for i in range(N):
    # Ti = fk_kuka2(q[i, :])
    Ti = robot.fk_casadi(q[i, :])
    opti.subject_to(xp[i] == Ti[0, 3])
    opti.subject_to(yp[i] == Ti[1, 3])
    opti.subject_to(zp[i] == Ti[2, 3])

# objective
V = ca.sum1(
    ca.sum2((q[:-1, :] - q[1:, :]) ** 2)
)  # + 0.05* ca.sumsqr(q) #+ 1 / ca.sum1(q[:, 4]**2)
opti.minimize(V)


opti.solver("ipopt")
opti.set_initial(q, q_inits[5])  # 2 3 4 5  converges
sol = opti.solve()

time_after = time.time()
run_time = time_after - time_before

qp_sol = opti.value(q)
cost = np.sum((qp_sol[:-1, :] - qp_sol[1:, :]) ** 2)
print("Cost: {}".format(cost))
print("Runtime: {}".format(run_time))

import matplotlib.pyplot as plt

fig, ax = get_default_axes3d([0, 1], [-0.5, 0.5], [0, 1])
ax.scatter(xp, yp, zp)
scene.plot(ax, c="g")
robot.animate_path(fig, ax, qp_sol)
ax.set_axis_off()
ax.view_init(24, 50)
plt.show()
