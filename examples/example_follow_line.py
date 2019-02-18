#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from acrobotics.util import get_default_axes3d
from acrobotics.recources.robots import Kuka
from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Shape, Collection

robot = Kuka()

path = []
for s in np.linspace(0, 1, 10):
    xi = 0.8
    yi = s * 0.2 + (1-s) * (-0.2)
    zi = 0.2
    path.append(FreeOrientationPt([xi, yi, zi]))

floor_plane = Shape(0.5, 0.5, 0.1)
floor_plane_tf = np.array([[1, 0, 0, 0.80],
                            [0, 1, 0, 0.00],
                            [0, 0, 1, 0.12],
                            [0, 0, 0, 1]])

scene = Collection([floor_plane], [floor_plane_tf])

#%% PLAN PATH
from acrobotics.planning import cart_to_joint_no_redundancy
from acrobotics.planning import get_shortest_path

qf_samples = np.linspace(-0.5, 0.5, 10)
Q = cart_to_joint_no_redundancy(robot, path, scene)

print([len(qi) for qi in Q])
qp = [qi[0] for qi in Q]

res = get_shortest_path(Q, method='dijkstra')
print(res)
qp_sol = res['path']


#%% ANIMATE

fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
for pi in path: pi.plot(ax2)
scene.plot(ax2, c='g')
robot.animate_path(fig2, ax2, qp_sol)
