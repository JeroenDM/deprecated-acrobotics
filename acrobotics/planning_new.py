import numpy as np
from typing import List, Callable
from enum import Enum
from .samplers import SampleMethod
from .geometry import Scene
from typing import List
from .path.path_pt import PathPt, TolEulerPt
from .path.joint_path import JointPath
from .cpp.graph import Graph
from .planning_setting import PlanningSetting
from .robot import Robot
from .path.path_pt import PathPt


class Planner:
    def __init__(
        self, robot: Robot, scene: Scene, path: List[PathPt], settings: PlanningSetting
    ):
        self.robot = robot
        self.scene = scene
        self.path = path
        self.settings = settings
        self.joint_path: JointPath = None
        self.first_step = True
        self.cost_function: Callable = None

    def step(self):
        if not self.first_step:
            self.reduce_tolerances()
        else:
            self.first_step = False

        joint_solutions = self.path_to_joint_solutions()
        self.find_shortest_joint_path(joint_solutions)

    def reset(self):
        self.first_step = False
        self.joint_path = None

    def path_to_joint_solutions(self):
        result = []
        for index, pt in enumerate(self.path):
            solutions = pt.to_joint_solutions(self.robot, self.settings, self.scene)
            if len(solutions) == 0:
                raise ValueError(f"PathPt {index} has no valid joint solutions.")
            else:
                print(f"Found {len(solutions)} joint solutions for PathPt {index}")
                result.append(solutions)
        return result

    def reduce_tolerances(self):
        for joint_sol, path_pt in zip(self.joint_path.joint_positions, self.path):
            fk_transform = self.robot.fk(joint_sol)
            path_pt.reduce_tolerance(
                fk_transform, self.settings.tolerance_reduction_factor
            )

    def find_shortest_joint_path(self, joint_solutions):
        if self.cost_function is not None:
            raise NotImplementedError

        joint_solutions_float32 = _check_dtype(joint_solutions)

        graph = Graph()
        for J in joint_solutions_float32:
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

        self.joint_path = JointPath(joint_path, cost)


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
        Q_float32 = []
        print("converting type of Q")
        for i in range(len(Q)):
            Q_float32.append(Q[i].astype("float32"))

        return Q_float32
    else:
        return Q
