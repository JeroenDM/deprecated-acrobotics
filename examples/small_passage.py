#!/usr/bin/env python3
import numpy as np

from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d, plot_reference_frame

from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Collection, Shape
from acrobotics.util import tf_inverse

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
si = path[0].get_samples(200, rep="quat")

for i, tp in enumerate(path):
    # si = tp.get_samples(100, rep="quat")
    row = []
    for qi in si:
        tfi = qi.transformation_matrix
        tfi[:-1, -1] += tp.p
        tfi = tfi @ goal_tf
        if not torch.is_in_collision(obstacles, tf_self=tfi):
            row.append(qi)
    print("Found {} cc free points for tp {}".format(len(row), i))
    data.append(row)


# samples = path[7].get_samples(200, rep="quat")
# for s in samples:
#     tfi = s.transformation_matrix @ goal_tf
#     tfi[:-1, -1] += path[7].p
#     # print(tfi)
#     if not torch.is_in_collision(obstacles, tf_self=tfi):
#         print("Found collision free tf {}".format(tfi[:-1, -1]))


sol = get_shortest_path(data)
print(sol)

# convert to transforms
sol_tf = []
for qi, tp in zip(sol["path"], path):
    sol_tf.append(qi.transformation_matrix)
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
