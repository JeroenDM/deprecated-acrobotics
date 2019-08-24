import numpy as np
from abc import ABC, abstractmethod
from typing import List
from .toleranced_number import (
    TolerancedNumber,
    FixedNumber,
    TolerancedQuaternion,
    IsToleranced,
)
from ..util import plot_reference_frame, create_grid, rpy_to_rotation_matrix
from ..samplers import Sampler, generate_quaternions, sample_SO3, SampleMethod
from ..robot import Robot
from ..types import SamplingType, SampleMethod
from ..geometry import Scene
from ..planning_setting import PlanningSetting
from ..pyquat_extended import QuaternionExtended as Quaternion


class PathPt(ABC):
    """
    Abstract base class for path points that a robot has to follow.

    Attributes
    ----------
    sampler : `~acrobotics.samplers.Sampler`
        Sampler to create random or deterministic samples.
    sample_dim : int
        Number of toleranced numbers for this path point.
    """

    sampler: Sampler
    values: List[IsToleranced]
    sample_dim: int

    @property
    def nominal_transform(self):
        return self.to_transform([value.nominal for value in self.values])

    @abstractmethod
    def to_transform(self, sampled_values) -> np.array:
        pass

    @abstractmethod
    def sample_grid(self) -> List[np.array]:
        pass

    @abstractmethod
    def sample_incremental(self, num_samples, method: SampleMethod) -> List[np.array]:
        pass

    @abstractmethod
    def calc_prev_value(self, fk_transform):
        pass

    def plot(self, ax, plot_frame=False):
        tf = self.nominal_transform
        ax.plot(tf[:3, 3], "o", c="r")
        if plot_frame:
            plot_reference_frame(ax, tf=tf)

    def __str__(self):
        # only print position
        return str(self.nominal_transform[:3, 3])

    def to_joint_solutions(
        self, robot: Robot, settings: PlanningSetting, scene: Scene = None
    ) -> np.ndarray:
        if settings.sampling_type == SamplingType.GRID:
            samples = self.sample_grid()
            joint_solutions = self._calc_ik(robot, samples)

        elif settings.sampling_type == SamplingType.INCREMENTAL:
            samples = self.sample_incremental(
                settings.num_samples, settings.sample_method
            )
            joint_solutions = self._calc_ik(robot, samples)

        elif settings.sampling_type == SamplingType.MIN_INCREMENTAL:
            joint_solutions = []
            for _ in range(settings.max_search_iters):
                samples = self.sample_incremental(
                    settings.step_size, settings.sample_method
                )
                temp_joint_solutions = self._calc_ik(robot, samples)
                for joint_position in temp_joint_solutions:
                    if not robot.is_in_collision(joint_position, scene):
                        joint_solutions.append(joint_position)
                if len(joint_solutions) >= settings.desired_num_samples:
                    return joint_solutions
            raise Exception("Maximum iterations reached in to_joint_solutions.")
        else:
            raise NotImplementedError
        collision_free_js = []
        for joint_position in joint_solutions:
            if not robot.is_in_collision(joint_position, scene):
                collision_free_js.append(joint_position)

        return np.array(collision_free_js)

    def reduce_tolerance(self, fk_transform, reduction_factor):
        previous_values = self.calc_prev_value(fk_transform)
        for prev_value, value in zip(previous_values, self.values):
            if value.is_toleranced:
                value.reduce_bounds(prev_value, reduction_factor)

    @staticmethod
    def _calc_ik(robot, samples) -> List:
        joint_solutions = []
        for transform in samples:
            ik_result = robot.ik(transform)
            if ik_result.success:
                joint_solutions.extend(ik_result.solutions)
        return joint_solutions

    @staticmethod
    def cast_path_values(values):
        result = []
        for number in values:
            if isinstance(number, TolerancedNumber):
                result.append(number)
            else:
                result.append(FixedNumber(number))
        return result

    @staticmethod
    def count_toleranced_values(values):
        return sum([isinstance(value, TolerancedNumber) for value in values])


class TolPositionPt(PathPt):
    def __init__(self, pos, quaternion: Quaternion):
        self.pos = super().cast_path_values(pos)
        self.values = self.pos
        self.quat = quaternion
        self.rotation_matrix = quaternion.rotation_matrix
        self.sample_dim = super().count_toleranced_values(self.values)
        self.sampler = Sampler()

    def sample_grid(self) -> List[np.array]:
        samples = create_grid([value.discretize() for value in self.values])
        return [self.to_transform(sample) for sample in samples]

    def sample_incremental(self, num_samples, method: SampleMethod) -> List[np.array]:
        # create a (num_samples x sample_dim) matrix with uniform samples
        R = self.sampler.sample(num_samples, self.sample_dim, method)

        # scale samples from range [0, 1] to desired range
        samples = np.zeros((num_samples, len(self.values)))
        cnt = 0
        for i, value in enumerate(self.values):
            if value.is_toleranced:
                samples[:, i] = R[:, cnt] * (value.upper - value.lower) + value.lower
                cnt += 1
            else:
                samples[:, i] = np.ones(num_samples) * value.nominal

        # convert position and euler angles to transforms
        return [self.to_transform(sample) for sample in samples]

    def to_transform(self, pos):
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_matrix
        transform[:3, 3] = pos
        return transform

    def calc_prev_value(self, fk_transform):
        return fk_transform[:3, 3]


class TolEulerPt(PathPt):
    """ Path point with fixed orientation and tol on position
    """

    def __init__(self, pos, rpy):
        self.pos = super().cast_path_values(pos)
        self.rpy = super().cast_path_values(rpy)
        self.values = self.pos + self.rpy
        self.sample_dim = super().count_toleranced_values(self.values)
        self.sampler = Sampler()

    def sample_grid(self) -> List[np.array]:
        samples = create_grid([value.discretize() for value in self.values])
        return [self.to_transform(sample) for sample in samples]

    def sample_incremental(self, num_samples, method: SampleMethod) -> List[np.array]:
        # create a (num_samples x sample_dim) matrix with uniform samples
        R = self.sampler.sample(num_samples, self.sample_dim, method)

        # scale samples from range [0, 1] to desired range
        samples = np.zeros((num_samples, len(self.values)))
        cnt = 0
        for i, value in enumerate(self.values):
            if value.is_toleranced:
                samples[:, i] = R[:, cnt] * (value.upper - value.lower) + value.lower
                cnt += 1
            else:
                samples[:, i] = np.ones(num_samples) * value.nominal

        # convert position and euler angles to transforms
        return [self.to_transform(sample) for sample in samples]

    def to_transform(self, values):
        T = np.eye(4)
        T[:3, 3] = values[:3]
        T[:3, :3] = rpy_to_rotation_matrix(values[3:])
        return T

    def calc_prev_value(self, fk_transform):
        raise NotImplementedError


class TolQuatPt(PathPt):
    """
    Orientation completely free, position is fixed.
    A nominal orientation is optional.
    """

    def __init__(self, pos, quaternion: TolerancedQuaternion):
        self.pos = np.array(pos)
        self.tol_quat = quaternion
        self.values = [self.tol_quat]
        self.sample_dim = 4  #  1 uniform and 3 gaussian TODO
        self.sampler = Sampler()

    def sample_grid(self):
        raise NotImplementedError

    def sample_incremental(self, num_samples, method: SampleMethod):
        if method == SampleMethod.random_uniform:
            samples = [
                self.tol_quat.quat.random_near(self.tol_quat.dist)
                for _ in range(num_samples)
            ]
            return [self.to_transform(quat) for quat in samples]
        else:
            raise NotImplementedError

    def to_transform(self, quat):
        transform = np.eye(4)
        transform[:3, :3] = quat.rotation_matrix
        transform[:3, 3] = self.pos
        return transform

    def calc_prev_value(self, fk_transform):
        return [Quaternion(matrix=fk_transform)]


# class FreeOrientationPt(TolQuatPt):
#     """
#     Orientation completely free, position is fixed.
#     A nominal orientation is optional.
#     """

#     def __init__(self, pos, nominal_quaternion=None):
#         self.pos = np.array(pos)
#         self.sample_dim = 3  #  fixed to generate uniform quaternions
#         self.sampler = Sampler()

#         if nominal_quaternion is not None:
#             raise NotImplementedError

#         self.prev_quaternion = None

#     def sample_grid(self):
#         raise NotImplementedError

#     def sample_incremental(self, num_samples, method: SampleMethod):
#         R = self.sampler.sample(num_samples, self.sample_dim, method)
#         quaternions = generate_quaternions(R)
#         return [self.to_transform(quat) for quat in quaternions]

#     def to_transform(self, quat):
#         transform = quat.transformation_matrix
#         transform[:3, 3] = self.pos
#         return transform

#     def calc_prev_value(self, fk_transform):
#         raise NotImplementedError

