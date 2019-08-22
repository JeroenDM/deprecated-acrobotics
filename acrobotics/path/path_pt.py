import numpy as np
from abc import ABC, abstractmethod
from ..pyquat_extended import QuaternionExtended as Quaternion
from .toleranced_number import TolerancedNumber, PathPointNumber
from ..util import plot_reference_frame, create_grid, rpy_to_rotation_matrix
from ..samplers import Sampler
from typing import List
from ..samplers import generate_quaternions, sample_SO3, SampleMethod


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

    @property
    def nominal_transform(self):
        return self.to_transform([value.nominal for value in self.values])

    def plot(self, ax, plot_frame=False):
        tf = self.nominal_transform
        ax.plot(tf[:3, 3], "o", c="r")
        if plot_frame:
            plot_reference_frame(ax, tf=tf)

    def __str__(self):
        # only print position
        return str(self.nominal_transform[:3, 3])

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

    @abstractmethod
    def to_transform(self, values) -> np.array:
        pass

    @staticmethod
    def cast_path_values(values):
        result = []
        for number in values:
            if isinstance(number, TolerancedNumber):
                result.append(number)
            else:
                result.append(PathPointNumber(number))
        return result

    @staticmethod
    def count_toleranced_values(values):
        return sum([isinstance(value, TolerancedNumber) for value in values])


class TolEulerPt(PathPt):
    """ Path point with fixed orientation and tol on position
    """

    def __init__(self, pos, rpy):
        super().__init__()
        self.pos = super().cast_path_values(pos)
        self.rpy = super().cast_path_values(rpy)
        self.values = self.pos + self.rpy
        self.sample_dim = super().count_toleranced_values(self.values)
        self.sampler = Sampler()

    def to_transform(self, values):
        T = np.eye(4)
        T[:3, 3] = values[:3]
        T[:3, :3] = rpy_to_rotation_matrix(values[3:])
        return T


class FreeOrientationPt(PathPt):
    """
    Orientation completely free, position is fixed.
    A nominal orientation is optional.
    """

    def __init__(self, pos, nominal_quaternion=None):
        self.pos = np.array(pos)
        self.sample_dim = 3  #  fixed to generate uniform quaternions
        self.sampler = Sampler()

        if nominal_quaternion is not None:
            raise NotImplementedError

    def sample_grid(self):
        raise NotImplementedError

    def sample_incremental(self, num_samples, method: SampleMethod):
        R = self.sampler.sample(num_samples, self.sample_dim, method)
        quaternions = generate_quaternions(R)
        return [self.to_transform(quat) for quat in quaternions]

    def to_transform(self, quat):
        transform = quat.transformation_matrix
        transform[:3, 3] = self.pos
        return transform


# class TolPositionPt(PathPt):
#     pass


# class AxisAnglePt(PathPt):
#     """ Trajectory point that has a tolerance on the end-effector orientation
#     given as +/- angle around an axis.
#     A tolerance on the position is also allowed.
#     """

#     def __init__(self, pos, axis, angle, q_nominal):
#         """ angle is toleranced number, the others are fixed
#         """
#         self.pos = super().cast_path_values(pos)
#         self.angle = super().cast_path_values([angle])[0]
#         self.values = self.pos + [self.angle]
#         self.quat = q_nominal
#         self.axis = axis
#         self.angle = angle

#     def to_transform(self, values):


#     def get_samples(self, num_samples, rep="transform", dist=None):
#         samples = []
#         for ai in self.angle.discretise():
#             qi = Quaternion(axis=self.axis, angle=ai) * self.quat
#             Ti = qi.transformation_matrix
#             Ti[:3, 3] = self.pos
#             samples.append(Ti)

#         return samples


# class FreeOrientationPt:
#     """ Trajectory point with fixed position and free orientation.
#     Work in progress
#     """

#     def __init__(self, position):
#         self.p = np.array(position)

#     def get_samples(self, num_samples, dist=None, **kwargs):
#         """ Sample orientation (position is fixed)
#         """
#         rep = "rpy"
#         if "rep" in kwargs:
#             rep = kwargs["rep"]

#         if rep == "rpy":
#             rpy = np.array(sample_SO3(n=num_samples, **kwargs))
#             pos = np.tile(self.p, (num_samples, 1))
#             return np.hstack((pos, rpy))
#         elif rep == "transform":
#             Ts = np.array(sample_SO3(n=num_samples, **kwargs))
#             for Ti in Ts:
#                 Ti[:3, 3] = self.p
#             return Ts
#         elif rep == "quat":
#             return sample_SO3(n=num_samples, **kwargs)
#         else:
#             raise ValueError("Invalid argument for rep: {}".format(rep))

#     def __str__(self):
#         return str(self.p)

#     def plot(self, ax):
#         ax.plot([self.p[0]], [self.p[1]], [self.p[2]], "o", c="r")


class TolOrientationPt:
    def __init__(self, position, orientation):
        self.p = np.array(position)
        self.o = orientation

    def get_samples(self, num_samples, rep=None, dist=0.1):
        """ sample near nominal orientation (position fixed)
        """
        samples = []
        for _ in range(num_samples):
            qr = Quaternion.random_near(self.o, dist)
            samples.append(qr.transformation_matrix)
        samples = np.array(samples)
        for Ti in samples:
            Ti[:3, 3] = self.p
        return samples

    def __str__(self):
        return str(self.p)

    def plot(self, ax):
        ax.plot([self.p[0]], [self.p[1]], [self.p[2]], "o", c="r")


class TolPositionPt:
    """ Path point with fixed orientation and tol on position
    """

    def __init__(self, pos, quat):
        self.pos = pos
        self.pos_has_tol, self.pos_nom = self._check_for_tolerance(pos)
        self.quat = quat

    def get_samples(self, num_samples, rep=None, dist=None):
        r = []
        # discretise position
        for i in range(3):
            if self.pos_has_tol[i]:
                r.append(self.pos[i].discretise())
            else:
                r.append(self.pos[i])

        grid = create_grid(r)

        samples = []
        for pi in grid:
            Ti = self.quat.transformation_matrix
            Ti[:3, 3] = pi
            samples.append(Ti)

        return samples

    def _check_for_tolerance(self, l):
        """ Check which value are toleranced numbers and get nominal values.
        Returns a list of booleans indication tolerance and a list of
        nominal values.
        """
        has_tolerance = [isinstance(num, TolerancedNumber) for num in l]
        nominal_vals = np.zeros(3)

        for i in range(3):
            if has_tolerance[i]:
                nominal_vals[i] = l[i].nominal
            else:
                nominal_vals[i] = l[i]

        return has_tolerance, nominal_vals

    def __str__(self):
        return str(self.pos)

    def plot(self, ax):
        ax.plot([self.pos[0]], [self.pos[1]], [self.pos[2]], "o", c="r")
