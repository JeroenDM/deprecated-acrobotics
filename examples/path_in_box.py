#!/usr/bin/env python3
"""
Follow a path in a box with tolerance on z-axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from acrobotics.resources.robots import KukaOnRail, Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.resources.workpiece_model import workpiece, path
from acrobotics.util import pose_x, pose_y
from acrobotics.util import plot_reference_frame, get_default_axes3d
from acrobotics.path import point_to_frame
from acrobotics.planning import cart_to_joint_simple, get_shortest_path

pi = np.pi

# robot definition
bot = KukaOnRail()
tf_base = np.dot(pose_x(-pi / 2, 0, -0.8, 0.5), pose_y(-pi / 2, 0, 0, 0))
bot.tf_base = tf_base
bot.tool = torch
q0 = [0, 0, 1.5, 0, 0, 0, 0]

bot_2 = Kuka()
bot_2.tool = torch

bot.do_check_self_collision = False

for pt in path:
    print(pt.rpy_nom)

from gui.task_editor import TaskEditor

gui = TaskEditor(bot_2, "/home/jeroen/Documents/github/deprecated-acrobotics/examples")
plt.show()


# plot situation
# fig, ax = get_default_axes3d()
# bot.plot(ax, q0, c="k")
# workpiece.plot(ax, c="g")
# for tp in path:
#     tf = point_to_frame(tp.p_nominal)
#     plot_reference_frame(ax, tf)
# plt.show(block=True)

# # plan path
# qf_samples = np.linspace(-0.5, 0.5, 10)
# Q = cart_to_joint_simple(bot, path, workpiece, qf_samples)

# print([len(qi) for qi in Q])
# # print(Q)

# w = np.array([1, 1, 1, 1, 1, 1, 1], dtype="float32")
# res = get_shortest_path(Q, method="dijkstra", weights=w)
# print(res)
# qp_sol = res["path"]

# # animate path
# fig2, ax2 = get_default_axes3d()
# ax2.set_axis_off()
# ax2.view_init(15, -30)
# for tp in path:
#     Ti = point_to_frame(tp.p_nominal)
#     plot_reference_frame(ax2, Ti)
# workpiece.plot(ax2, c="g")
# bot.animate_path(fig2, ax2, qp_sol)

# # bot.animation.save('../box_path.gif', writer='imagemagick', fps=5)

# plt.show(block=True)
