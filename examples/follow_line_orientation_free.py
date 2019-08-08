#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from acrobotics.util import get_default_axes3d
from acrobotics.resources.robots import Kuka
from acrobotics.io import load_task, load_settings

robot = Kuka()
path, scene = load_task("examples/line_orient_free.json")
settings = load_settings("examples/planner_settings_simple.json")

#%% PLAN PATH
from acrobotics.planning import cart_to_joint_no_redundancy
from acrobotics.planning import get_shortest_path

Q = cart_to_joint_no_redundancy(robot, path, scene, num_samples=settings["num_samples"])

print([len(qi) for qi in Q])
qp = [qi[0] for qi in Q]

res = get_shortest_path(Q, method=settings["graph_search_method"])
print(res)
qp_sol = res["path"]


#%% ANIMATE
import matplotlib.pyplot as plt

fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
for pi in path:
    pi.plot(ax2)
scene.plot(ax2, c="g")
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
