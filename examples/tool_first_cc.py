#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from acrobotics.util import get_default_axes3d, pose_x, plot_reference_frame
from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Shape, Scene
from acrobotics.planning import cart_to_joint_tool_first_cc
from acrobotics.planning import cart_to_joint_no_redundancy
from acrobotics.planning import get_shortest_path

robot = Kuka()
# robot.set_tool(torch)

path = []
for s in np.linspace(0, 1, 5):
    xi = 0.8
    yi = s * 0.2 + (1 - s) * (-0.2)
    zi = 0.2
    path.append(FreeOrientationPt([xi, yi, zi]))

floor_plane = Shape(0.5, 0.5, 0.1)
floor_plane_tf = pose_x(0.0, 0.8, 0.0, 0.0)

scene = Scene([floor_plane], [floor_plane_tf])
# scene = Scene([], [])

fig, ax = get_default_axes3d()
q0 = [0, 1.5, 0, 0, 0, 0]
robot.plot(ax, q0, c="k")
scene.plot(ax, c="g")
for tp in path:
    tp.plot(ax)

tf_tt = robot.fk(q0)
plot_reference_frame(ax, tf_tt)

# PLAN PATH
Q = cart_to_joint_no_redundancy(robot, path, scene)
# Q = cart_to_joint_tool_first_cc(robot, path, scene)
# Q = [[], []]
print([len(qi) for qi in Q])

if not np.all(np.array([len(qi) for qi in Q])):
    print("No path found")
    plt.show(block=True)
    exit()

res = get_shortest_path(Q, method="dijkstra")

print(res)
qp_sol = res["path"]


# ANIMATE

fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
for pi in path:
    pi.plot(ax2)
scene.plot(ax2, c="g")
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
