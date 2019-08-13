import matplotlib.pyplot as plt
import numpy as np


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from acrobotics.resources.robots import Kuka
from acrobotics.util import get_default_axes3d
from acrobotics.geometry import Shape, Collection

robot = Kuka()

# fig, ax = get_default_axes3d()
# robot.plot(ax, [0, 1.5, 0, 0, 0, 0], c="k")
# fig.show()


def check_voxel(robot, q, pos):
    ds = 0.1
    tf_box = np.eye(4)
    tf_box[:3, 3] = pos
    box = Collection([Shape(ds, ds, ds)], [tf_box])

    return robot.is_in_collision(q, box)

def scale_point(pos):
    return pos / (N - 1)

def check_path(robot, path, pos):
    for qi in path:
        res = check_voxel(robot, qi, pos)
        if res:
            return True


N = 10
x, y, z = np.indices((N, N, N))
voxels = np.zeros((N, N, N), dtype=bool)

q_check = [0.5, 1.5, 0, 0, 0, 0]
q_path = [[0.0, 1.5, 0, 0, 0, 0],
          [0.3, 1.5, 0, 0, 0, 0],
          [0.6, 1.5, 0, 0, 0, 0],
          [0.9, 1.5, 0, 0, 0, 0]]

points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
for pi in points:
    #voxels[pi[0], pi[1], pi[2]] = check_voxel(robot, q_check, scale_point(pi))
    voxels[pi[0], pi[1], pi[2]] = check_path(robot, q_path, scale_point(pi))

# set the colors of each object
colors = np.empty(voxels.shape, dtype=object)
colors[voxels] = "red"

# and plot everything
fig = plt.figure()
ax = fig.gca(projection="3d")
robot.plot(ax, q_check, c="k")

fig2 = plt.figure()
ax2 = fig2.gca(projection="3d")
ax2.voxels(voxels, facecolors=colors, edgecolor="k")

plt.show()
