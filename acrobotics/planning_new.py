import numpy as np
from typing import List, Callable
from enum import Enum
from .samplers import SampleMethod
from .geometry import Scene
from typing import List
from .path.path_pt import PathPt, FreeOrientationPt, TolEulerPt
from .path.joint_path import JointPath
from .cpp.graph import Graph
from .planning_setting import PlanningSetting
from .robot import Robot
from .path.path_pt import PathPt


def path_to_joint_solutions(
    path: List[PathPt], robot: Robot, settings: PlanningSetting, scene: Scene
):
    result = []
    for index, pt in enumerate(path):
        solutions = pt.to_joint_solutions(robot, settings, scene)
        if len(solutions) == 0:
            raise ValueError(f"PathPt {index} has no valid joint solutions.")
        else:
            print(f"Found {len(solutions)} joint solutions for PathPt {index}")
            result.append(solutions)
    return result


def reduce_tolerance(path_pt: PathPt, joint_position: np.ndarray) -> PathPt:
    return path_pt.reduce_tolerance(joint_position)


def find_shortest_joint_path(
    joint_solutions: List[np.ndarray], cost_function: Callable = None
):
    if cost_function is not None:
        raise NotImplementedError

    joint_solutions = _check_dtype(joint_solutions)

    graph = Graph()
    for J in joint_solutions:
        graph.add_data_column(J)
    graph.init()
    graph.run_dijkstra()

    joint_path_indices = graph.get_path(len(joint_solutions))
    cost = graph.get_path_cost()

    # if graph search fails, the first path point is set to -1
    # this shouldn't happen
    assert joint_path_indices[0] != -1

    joint_path = []
    for js, index in zip(joint_solutions, joint_path_indices):
        joint_path.append(js[index])

    return JointPath(joint_path, cost)


def _check_dtype(Q):
    """ Change type if necessary to float32

    Due to an unresolved issue with swig and numpy, I have to convert the type.

    Parameters
    ----------
    Q : list of nympy.ndarrays of float
        A list with the possible joint positions for every trajectory point
        along a path.

    Returns
    -------
    list of nympy.ndarrays of float32
    """
    if Q[0].dtype != "float32":
        print("converting type of Q")
        for i in range(len(Q)):
            Q[i] = Q[i].astype("float32")

    return Q
