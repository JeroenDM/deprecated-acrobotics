from acrobotics.planning_new import Planner
from acrobotics.types import SamplingType
from acrobotics.types import SampleMethod
from acrobotics.planning_setting import PlanningSetting
from acrobotics.geometry import Shape, Scene
from acrobotics.path.path_pt import TolQuatPt
from acrobotics.path.toleranced_number import TolerancedQuaternion
from acrobotics.pyquat_extended import QuaternionExtended as Quaternion

import numpy as np
import matplotlib.pyplot as plt

from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d

path_ori_free = []
for s in np.linspace(0, 1, 3):
    xi = 0.8
    yi = s * 0.2 + (1 - s) * (-0.2)
    zi = 0.2
    path_ori_free.append(
        TolQuatPt([xi, yi, zi], TolerancedQuaternion(Quaternion(), 1.0))
    )

table = Shape(0.5, 0.5, 0.1)
table_tf = np.array([[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]])
scene1 = Scene([table], [table_tf])

robot = Kuka()
# robot.tool = torch

settings = PlanningSetting(
    SamplingType.INCREMENTAL,
    SampleMethod.random_uniform,
    500,
    tolerance_reduction_factor=2,
)

print("Quat dist: {}".format(path_ori_free[0].values[0].dist))
print("Quat: {}".format(path_ori_free[0].values[0].quat))

planner = Planner(robot, scene1, path_ori_free, settings)
planner.step()
planner.step()

print("Quat dist: {}".format(path_ori_free[0].values[0].dist))
print("Quat: {}".format(path_ori_free[0].values[0].quat))

# for i in range(3):
#     planner.step()
#     joint_path = planner.joint_path

#     fig, ax = get_default_axes3d()
#     scene1.plot(ax, c="g")
#     robot.animate_path(fig, ax, joint_path.joint_positions)
#     plt.show(block=True)
