#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from acrobotics.util import get_default_axes3d,plot_reference_frame
from acrobotics.recources.robots import Kuka
from acrobotics.path import TolPositionPoint, TolerancedNumber
from acrobotics.geometry import Shape, Collection
from pyquaternion import Quaternion

robot = Kuka()

path = []

xt = TolerancedNumber(0.4, 1.2)
q1 = Quaternion(axis=[0, 1, 0], angle=np.pi)

for s in np.linspace(0, 1, 8):
    yi = s * 0.2 + (1-s) * (-0.2)
    zi = 0.2
    path.append(TolPositionPoint([xt, yi, zi], q1))

# print(path[0].discretise())

# fig, ax = get_default_axes3d()
# for Ti in path[0].discretise():
#     plot_reference_frame(ax, tf=Ti)
#
# for Ti in path[5].discretise():
#     plot_reference_frame(ax, tf=Ti)
# plt.show()

floor_plane = Shape(0.5, 0.5, 0.1)
floor_plane_tf = np.array([[1, 0, 0, 0.80],
                            [0, 1, 0, 0.00],
                            [0, 0, 1, 0.12],
                            [0, 0, 0, 1]])

obstacle = Shape(0.1, 0.1, 0.5)
obstacle_tf = np.array([[1, 0, 0, 1.00],
                            [0, 1, 0, 0.25],
                            [0, 0, 1, 0.12],
                            [0, 0, 0, 1]])

scene = Collection([floor_plane, obstacle],
                    [floor_plane_tf, obstacle_tf])

#%% PLAN PATH
from acrobotics.planning import cart_to_joint_iterative,cart_to_joint_no_redundancy
from acrobotics.planning import get_shortest_path
#res = cart_to_joint_iterative(robot, path, scene, num_samples=200, max_iters=10)
Q = cart_to_joint_no_redundancy(robot, path, scene, num_samples=200)

res = get_shortest_path(Q, method='dijkstra')
print(res)
qp_sol = res['path']


#%% ANIMATE
import matplotlib.pyplot as plt
fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
for pi in path: pi.plot(ax2)
scene.plot(ax2, c='g')
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
