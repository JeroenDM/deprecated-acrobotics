#!/usr/bin/env python3
import numpy as np
from pyquaternion import Quaternion

from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d, plot_reference_frame

from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Collection, Shape
from acrobotics.util import tf_inverse

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

# get orientation samples for each path point
from acrobotics.pygraph import get_shortest_path


data = []
# notice how having the same orientation samples
# for all path points we can have a path of length 0
# if there are no obstacles
# si = path[0].get_samples(200, rep="quat")

for i, tp in enumerate(path):
    si = tp.get_samples(2000, rep="quat")
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

# print("State data matrices")
# print([d.shape for d in data])


costs = []
for i in range(1, len(data)):
    ci = data[i - 1] @ data[i].T
    ci = np.arccos(np.minimum(np.abs(ci), 1.0))
    costs.append(ci)

# print("Cost matrices:")
# print([c.shape for c in costs])


# caluclate shortest path
state_dims = [len(d) for d in data]
values = [np.zeros(si) for si in state_dims]
indices = [np.zeros(si, dtype=int) for si in state_dims]

print(state_dims)

for i in range(len(values) - 2, -1, -1):
    # print("Iteration: {}".format(i))
    c_current = costs[i] + values[i + 1]
    values[i] = np.min(c_current, axis=1)
    indices[i] = np.argmin(c_current, axis=1)

# for v in values:
#     print(v)
# for id in indices:
#     print(id)

# find shortest path
sol = []
i_shortest_dist = np.argmin(values[0])
sol.append(data[0][i_shortest_dist])
i_next = int(indices[0][i_shortest_dist])

for i in range(len(data)):
    sol.append(data[i][i_next])
    i_next = int(indices[i][i_next])

# print(sol)

#
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
    if i % 4 == 0:
        tfi = tf @ goal_tf
        # if not torch.is_in_collision(obstacles, tf_self=tfi):
        torch.plot(ax, c="r", tf=tfi)


fig.show()
