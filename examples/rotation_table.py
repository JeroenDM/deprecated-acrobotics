#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from acrobotics.robot import Robot, DHLink, Link, Tool
from acrobotics.geometry import Shape, Collection
import matplotlib.pyplot as plt

from acrobotics.resources.workpiece_model import workpiece, path, path2, path3
from acrobotics.util import plot_reference_frame, get_default_axes3d
from acrobotics.util import pose_y, pose_x
from acrobotics.path import point_to_frame
from acrobotics.resources.robots import KukaOnRail
from acrobotics.resources.torch_model import torch
from acrobotics.planning import get_shortest_path

pi = np.pi

# robot definition
bot = KukaOnRail()
tf_base = np.dot(pose_x(-pi / 2, 0, -0.8, 0.5), pose_y(-pi / 2, 0, 0, 0))
bot.tf_base = tf_base
bot.tool = torch

q0 = [0, 0, 1.5, 0, 0, 0, 0]

l = Link(DHLink(0, 0, 0, 0), "r",  Collection([Shape(0.1, 0.1, 0.02)], [pose_y(np.pi / 2, 0.01, 0, 0)]))

table = Robot([l])
table.tf_base = pose_y(np.pi / 2, 0, 0, 0)
table.tool = Tool([], [], pose_y(-np.pi / 2, 0, 0, 0))

q_table = [1.0]
tee = table.fk(q_table)

    
def cart_to_joint_simple(robot, path, scene, q_fixed, table, q_table):
    """ cartesian path to joint solutions
    
    q_fixed argument provides an already sampled version
    for the redundant joints.
    The path tolerance is sampled by tp.discretise inside
    the TrajectoryPoint class.
    
    Return array with float32 elements, reducing data size.
    During graph search c++ floats are used, also float32.
    """
    Q = []
    
    for i, tp in enumerate(path):
        print("Processing point " + str(i + 1) + "/" + str(len(path)))
        for Ti in tp.discretise():
            q_sol = []
            for q_table_i in q_table:
                Ti = table.fk(q_table_i) @ Ti
                for qfi in q_fixed:
                    sol = robot.ik(Ti, qfi)
                    if sol["success"]:
                        for qi in sol["sol"]:
                            if not robot.is_in_collision(qi, scene):
                                q_sol.append(qi)
        print("Found {} valid configs for point {}".format(len(q_sol), i))
        if len(q_sol) > 0:
            Q.append(np.vstack(q_sol).astype("float32"))
        else:
            Q.append([])
    return Q

# qf_samples = np.linspace(-0.5, 0.5, 10)
# qt_samples = [[qi] for qi in np.linspace(0, 1.0, 5)]
# Q = cart_to_joint_simple(bot, path3, workpiece, qf_samples, table, qt_samples)

# w = np.array([1, 1, 1, 1, 1, 1, 1], dtype="float32")
# res = get_shortest_path(Q, method="dijkstra", weights=w)
# print(res)
# qp_sol = res["path"]



# plot situation
fig, ax = get_default_axes3d()
# bot.plot(ax, q0, c='k')
table.plot(ax, q_table, c="k")
workpiece.plot(ax, tf=tee, c='g')
for tp in path:
    Ti = tee @ point_to_frame(tp.p_nominal)
    plot_reference_frame(ax, Ti)
    
for tp in path2:
    Ti = tee @ point_to_frame(tp.p_nominal)
    plot_reference_frame(ax, Ti)

for tp in path3:
    Ti = tee @ point_to_frame(tp.p_nominal)
    plot_reference_frame(ax, Ti)

# bot.animate_path(fig, ax, qp_sol)

plt.show(block=True)