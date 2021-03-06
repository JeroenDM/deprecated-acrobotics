#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.resources.robots import Kuka
from acrobotics.resources.path_on_table import (
    path_pos_tol,
    path_ori_free,
    scene2,
    scene1,
)
from pyquaternion import Quaternion

robot = Kuka()
# path = path_pos_tol
path = path_ori_free

from acrobotics.planning import cart_to_joint_iterative

res = cart_to_joint_iterative(robot, path, scene1, num_samples=500, max_iters=5)
# print(res)

print(res["costs"])
print(res["times"])
res["N"] = len(path)

import numpy

numpy.save("res_500_case_1", res)

qp_sol = res["path"]

import matplotlib.pyplot as plt
from acrobotics.util import get_default_axes3d, plot_reference_frame

fig, ax = plt.subplots()
ax.plot(res["costs"], "o-")
ax.set_title("Cost as function of iterations")

fig2, ax2 = get_default_axes3d([0, 1], [-0.5, 0.5], [0, 1])
for pi in path:
    pi.plot(ax2)
scene1.plot(ax2, c="g")
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
