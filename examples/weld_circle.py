#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from numpy import cos, sin, pi
from pyquaternion import Quaternion
from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.resources.robots import Kuka
from acrobotics.io import load_task


from acrobotics.resources.torch_model import torch

robot = Kuka()
robot.tool = torch

# task = load_task("examples/weld_circle.json")
task = load_task("examples/small_passage_2.json")


q0 = [0, np.pi / 2, 0, 0, 0, 0]
fig, ax = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
robot.plot(ax, q0, c="k")
task.plot(ax)
plot_reference_frame(ax, tf=robot.fk(q0))
#
# plot_reference_frame(ax, tf=q1.transformation_matrix)
#
# plt.show()
#
# fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
# for pi in path: pi.plot(ax2)
#
# for Ti in path[0].get_samples(None):
#     plot_reference_frame(ax2, tf=Ti)
#
# for Ti in path[3].get_samples(None):
#     plot_reference_frame(ax2, tf=Ti)
#
# plt.show()

# # table with obstacle
# table = Shape(0.5, 0.5, 0.1)
# table_tf = np.array([[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.02], [0, 0, 0, 1]])
#
# obstacle = Shape(0.1, 0.1, 0.5)
# obstacle_tf = np.array([[1, 0, 0, 1.0], [0, 1, 0, 0.15], [0, 0, 1, 0.12], [0, 0, 0, 1]])
#
# # scene = Scene([table, obstacle],
# #                    [table_tf, obstacle_tf])
# scene = Scene([table], [table_tf])
# # scene = Scene([], [])

#%% PLAN PATH
# from acrobotics.planning import cart_to_joint_no_redundancy
# from acrobotics.planning import get_shortest_path
# from acrobotics.planning import cart_to_joint_iterative
#
# res = cart_to_joint_iterative(
#     robot, task.path, task.scene, num_samples=200, max_iters=3
# )

res = task.solve(robot, method="grid_incremental")

# Q = cart_to_joint_no_redundancy(robot, path, scene, num_samples=1000)
# print([len(qi) for qi in Q])
# res = get_shortest_path(Q, method="dijkstra")


qp_sol = res["path"]

# from acrobotics.planning import cart_to_joint_iterative
#
# res = cart_to_joint_iterative(robot, path, scene, num_samples=20, max_iters=5)
# print(res)
# qp_sol = res["path"]
# #
# #%% ANIMATE
# fig, ax = plt.subplots()
# ax.plot(res["costs"], "o-")
# ax.set_title("Cost as function of iterations")
#
# import matplotlib.pyplot as plt
#
fig2, ax2 = get_default_axes3d([0, 1.5], [-0.75, 0.75], [0, 1.5])
task.plot(ax2)
robot.animate_path(fig2, ax2, qp_sol)
ax2.set_axis_off()

# robot.animation.save('../animation.gif', writer='imagemagick', fps=5)

plt.show(block=True)
