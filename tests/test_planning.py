import numpy as np

from acrobotics.planning import *
from acrobotics.geometry import Shape, Collection
from acrobotics.path import FreeOrientationPt
from acrobotics.util import get_default_axes3d


def test_planning_task():
    fig, ax = get_default_axes3d()
    scene = Collection([Shape(0.1, 0.1, 0.1)], [np.eye(4)])
    path = [FreeOrientationPt([i, i, i]) for i in range(5)]
    task = PlanningTask(path, scene)
    assert task.has_obstacles
    task.plot(ax)

    task = PlanningTask(path)
    assert not task.has_obstacles
    task.plot(ax)
