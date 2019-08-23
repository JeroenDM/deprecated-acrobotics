from acrobotics.planning_new import path_to_joint_solutions, find_shortest_joint_path
from acrobotics.types import SamplingType
from acrobotics.types import SampleMethod
from acrobotics.planning_setting import PlanningSetting
from acrobotics.path.path_pt import FreeOrientationPt
from acrobotics.geometry import Shape, Scene

import numpy as np
import matplotlib.pyplot as plt

from acrobotics.resources.robots import Kuka
from acrobotics.resources.torch_model import torch
from acrobotics.util import get_default_axes3d


class TestPlanning:
    def test_complete_problem(self):
        path_ori_free = []
        for s in np.linspace(0, 1, 3):
            xi = 0.8
            yi = s * 0.2 + (1 - s) * (-0.2)
            zi = 0.2
            path_ori_free.append(FreeOrientationPt([xi, yi, zi]))

        table = Shape(0.5, 0.5, 0.1)
        table_tf = np.array(
            [[1, 0, 0, 0.80], [0, 1, 0, 0.00], [0, 0, 1, 0.12], [0, 0, 0, 1]]
        )
        scene1 = Scene([table], [table_tf])

        robot = Kuka()
        # robot.tool = torch

        settings = PlanningSetting(
            SamplingType.INCREMENTAL, SampleMethod.random_uniform, 500
        )

        joint_positions = path_to_joint_solutions(
            path_ori_free, robot, settings, scene1
        )
        joint_path = find_shortest_joint_path(joint_positions)

        fig, ax = get_default_axes3d()
        scene1.plot(ax, c="g")
        robot.animate_path(fig, ax, joint_path.joint_path)

