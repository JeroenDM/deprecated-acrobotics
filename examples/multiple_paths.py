import numpy as np

from acrobotics.io import load_task
from acrobotics.util import get_default_axes3d
from acrobotics.dp import shortest_path
from acrobotics.planning import cart_to_joint_no_redundancy

task = load_task("examples/line_x_tol.json")


#%% ANIMATE
import matplotlib.pyplot as plt

fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
task.plot(ax2)

plt.show(block=True)
