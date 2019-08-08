#!/usr/bin/env python3
import numpy as np
from pyquaternion import Quaternion

from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.util import tf_inverse
from acrobotics.dp import *
from acrobotics.io import load_task

## TODO:
# is squared cost better than L1 norm,
# I would expect it is to avoid jumps.
path, obstacles = load_task("examples/small_passage.json")

# goal frame to plot tool
goal_tf = tf_inverse(torch.tf_tt)

data = []
# notice how having the same orientation samples
# for all path points we can have a path of length 0
# if there are no obstacles
# and even if there are obstacles
# => there is potentially a very fast way to solve this problem
# using a convex of of the start and end point
# si = path[0].get_samples(2000, rep="quat")

for i, tp in enumerate(path):
    si = tp.get_samples(1000, rep="quat")
    row = []
    for qi in si:
        tfi = qi.transformation_matrix
        tfi[:-1, -1] += tp.p
        tfi = tfi @ goal_tf
        if not torch.is_in_collision(obstacles, tf_self=tfi):
            row.append(qi.q)
    print("Found {} cc free points for tp {}".format(len(row), i))
    data.append(row)

data = [np.array(d) for d in data]


def cost_function(C1, C2):
    """ Input size should be N x 4.
    N is the number of points to compare, 4 is the number of quaternion elements
    for this cost functions.
    """
    ci = C1 @ C2.T
    ci = np.arccos(np.minimum(np.abs(ci), 1.0))
    return ci


def cost_function_2(C1, C2):
    """ Alternative cost functions that also takes into account the distance
    from a given goal orientation.
    """
    q_ref = Quaternion(axis=[0, 1, 0], angle=1.5).q
    C_ref = np.tile(q_ref, (len(C2), 1))
    ci = C1 @ C2.T
    path_cost = np.arccos(np.minimum(np.abs(ci), 1.0))

    # path_cost[path_cost > 0.2] = np.inf

    cj = C1 @ C_ref.T
    state_cost = np.arccos(np.minimum(np.abs(cj), 1.0))
    return path_cost ** 2 + 0.5 * state_cost ** 2


# costs = apply_cost_function(data, cost_function)
# indices, values = calculate_value_function(costs)
# sol = extract_shortest_path(data, indices, values)
sol = shortest_path(data, cost_function)

# convert to transforms
sol_tf = []
for qi, tp in zip(sol["path"], path):
    sol_tf.append(Quaternion(qi).transformation_matrix)
    sol_tf[-1][:-1, -1] = tp.p


# import matplotlib.pyplot as plt
# plt.matshow(costs[0])
# plt.show()

# leg = []
# i = 0
# for v in values:
#     t = np.linspace(0, 1, len(v))
#     # plt.plot(t, np.sort(v))
#     plt.plot(t, v)
#     leg.append(i)
#     i += 1
# plt.legend(leg)
# plt.show()

fig, ax = get_default_axes3d()
plot_reference_frame(ax)
# torch.plot(ax, c="k", tf=goal_tf)
obstacles.plot(ax, c="g")
for tp in path:
    tp.plot(ax)

for i, tf in enumerate(sol_tf):
    if i % 2 == 0:
        tfi = tf @ goal_tf
        # if not torch.is_in_collision(obstacles, tf_self=tfi):
        torch.plot(ax, c="r", tf=tfi)

fig.show()
