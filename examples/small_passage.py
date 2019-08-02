#!/usr/bin/env python3
import numpy as np
from pyquaternion import Quaternion

from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d, plot_reference_frame

from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Collection, Shape
from acrobotics.util import tf_inverse
from acrobotics.dp import *

## TODO:
# is squared cost better than L1 norm,
# I would expect it is to avoid jumps.

# define path
path = []
Np = 15
yr = np.linspace(-0.1, 0.1, Np)
for i in range(Np):
    x, y, z = 0.0, yr[i], 0.0
    path.append(FreeOrientationPt([x, y, z]))

# define obstacles
gap = 0.005
offset = (0.025 + 0.1) / 2 + gap
shapes = [
    Shape(0.1, 0.03, 0.2),
    Shape(0.1, 0.03, 0.2),
    Shape(0.025 + 2 * gap, 0.03, 0.1),
]
shape_tfs = [np.eye(4), np.eye(4), np.eye(4)]
shape_tfs[0][:-1, -1] = np.array([-offset, 0, 0])
shape_tfs[1][:-1, -1] = np.array([offset, 0, 0])
shape_tfs[2][:-1, -1] = np.array([0, 0, 0.05])
obstacles = Collection(shapes, shape_tfs)

# goal frame to plot tool
goal_tf = tf_inverse(torch.tf_tt)

data = []
# notice how having the same orientation samples
# for all path points we can have a path of length 0
# if there are no obstacles
# and even if there are obstacles
# => there is potentially a very fast way to solve this problem
# using a convex of of the start and end point
si = path[0].get_samples(2000, rep="quat")

for i, tp in enumerate(path):
    # si = tp.get_samples(2000, rep="quat")
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

    path_cost[path_cost > 0.2] = np.inf

    cj = C1 @ C_ref.T
    state_cost = np.arccos(np.minimum(np.abs(cj), 1.0))
    return path_cost ** 2 + 0.5 * state_cost ** 2


costs = apply_cost_function(data, cost_function_2)
indices, values = calculate_value_function(costs)
sol = extract_shortest_path(data, indices, values)

# convert to transforms
sol_tf = []
for qi, tp in zip(sol, path):
    sol_tf.append(Quaternion(qi).transformation_matrix)
    sol_tf[-1][:-1, -1] = tp.p


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
