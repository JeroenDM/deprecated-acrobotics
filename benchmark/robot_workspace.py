import matplotlib.pyplot as plt
import numpy as np


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from acrobotics.resources.robots import Kuka
from acrobotics.util import get_default_axes3d, plot_reference_frame
from acrobotics.geometry import Shape, Collection
from acrobotics.util import sample_SO3


robot = Kuka()

N = 10
x, y, z = np.indices((N, N, N))
voxels = np.zeros((N, N, N), dtype=bool)

points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

def scale_point(pos):
    return pos / (N - 1) * 2 - 1

def sample_position(p, n = 5):
    tf_samples = sample_SO3(n=n, rep="transform")
    for tfi in tf_samples:
        tfi[:3, 3] = p
    return tf_samples

def process_ik_solution(ik_solution):
    if ik_solution['success']:
        return len(ik_solution['sol'])
    else:
        return 0


def get_reachability(robot, p):
    NUM_CONFIGS = 8  # the robot has at most 8 different ik solutions
    tf_samples = sample_position(p)
    count = 0
    for tfi in tf_samples:
        count += process_ik_solution(robot.ik(tfi))
    return count / (NUM_CONFIGS * len(tf_samples))
        

reachability = np.zeros((N, N, N))

for pi in points:
    #voxels[pi[0], pi[1], pi[2]] = check_voxel(robot, q_check, scale_point(pi))
    reachability[pi[0], pi[1], pi[2]] = get_reachability(robot, scale_point(pi))


np.save('reachability_kuka_{}.npy'.format(N), reachability)
#%%
#voxels = reachability > 0.5
voxels = reachability > 0.2

cmap = plt.cm.RdYlBu
colors = cmap(reachability)
colors[:, :, :, -1] = 0.9  #  make transparant

fig2 = plt.figure()
ax2 = fig2.gca(projection="3d")
ax2.set_axis_off()
ax2.voxels(voxels, facecolors=colors, edgecolor="k")