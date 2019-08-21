#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from acrobotics.util import get_default_axes3d
from acrobotics.resources.robots import Kuka
from acrobotics.io import load_task, load_settings

robot = Kuka()
task = load_task("examples/line_orient_free.json")
settings = load_settings("examples/planner_settings_simple.json")

qp_init = np.array([0.3, 0, 0, 0, 0, 0])
task.set_initial_path(qp_init)
task.opt_based_max_iters = 1000

res = task.solve(robot, method="grid_incremental")

solver = create_solver("gridIncremental")
solver(robot, task)


qp_sol = res["path"]


#%% ANIMATE
fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
task.plot(ax2)
robot.animate_path(fig2, ax2, qp_sol)
plt.show(block=True)
