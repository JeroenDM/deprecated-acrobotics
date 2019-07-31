#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt
from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.resources.robots import Kuka
from acrobotics.path import AxisAnglePt, TolerancedNumber, FreeOrientationPt
from acrobotics.geometry import Shape, Collection
from pyquaternion import Quaternion

from acrobotics.resources.torch_model import torch

robot = Kuka()
robot.tool = torch


# q1 = Quaternion(axis=[1, 0, 0], angle=np.pi)
R1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
q1 = Quaternion(matrix=R1)

at = TolerancedNumber(-np.pi / 2, np.pi / 2, samples=20)
tilt_angle = np.pi / 4
nom_axis = np.array([0, 0, 1])
path = []
for s in np.linspace(0, 1, 20):
    # xi = 0.6
    # yi = s * 0.2 + (1-s) * (-0.2)
    # zi = 0.2

    angle = 2 * pi * s
    xi = 0.8 - 0.2 * cos(angle)
    yi = -0.2 * sin(angle)
    zi = 0.1
    axisi = [sin(angle), -cos(angle), 0]
    q_tilt = Quaternion(axis=axisi, angle=tilt_angle)
    qi = q_tilt * q1
    tol_axis_i = np.dot(qi.rotation_matrix, nom_axis)
    path.append(AxisAnglePt([xi, yi, zi], tol_axis_i, at, qi))

    # path.append(FreeOrientationPt([xi, yi, zi]))


# q0 = [0, np.pi/2, 0, 0, 0, 0]
# fig, ax = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
# robot.plot(ax, q0, c='b')
# plot_reference_frame(ax, tf=robot.fk(q0))
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

# table with obstacle
table = Shape(0.5, 0.5, 0.1)
table_tf = np.array([[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.02], [0, 0, 0, 1]])

obstacle = Shape(0.1, 0.1, 0.5)
obstacle_tf = np.array([[1, 0, 0, 1.0], [0, 1, 0, 0.15], [0, 0, 1, 0.12], [0, 0, 0, 1]])

# scene = Collection([table, obstacle],
#                    [table_tf, obstacle_tf])
scene = Collection([table], [table_tf])
# scene = Collection([], [])

#%% PLAN PATH
# from acrobotics.planning import cart_to_joint_no_redundancy
# from acrobotics.planning import get_shortest_path
#
# Q = cart_to_joint_no_redundancy(robot, path, scene)
#
# print([len(qi) for qi in Q])
# qp = [qi[0] for qi in Q]
#
# res = get_shortest_path(Q, method='dijkstra')
# print(res)
# qp_sol = res['path']

from acrobotics.planning import cart_to_joint_iterative

res = cart_to_joint_iterative(robot, path, scene, num_samples=20, max_iters=5)
print(res)
qp_sol = res["path"]

#%% ANIMATE
fig, ax = plt.subplots()
ax.plot(res["costs"], "o-")
ax.set_title("Cost as function of iterations")

import matplotlib.pyplot as plt

fig2, ax2 = get_default_axes3d([0, 1.5], [-0.75, 0.75], [0, 1.5])
for pi in path:
    pi.plot(ax2)
scene.plot(ax2, c="g")
robot.animate_path(fig2, ax2, qp_sol)
ax2.set_axis_off()

# robot.animation.save('../animation.gif', writer='imagemagick', fps=5)

plt.show(block=True)
