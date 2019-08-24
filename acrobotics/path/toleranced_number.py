"""
Tools to specify a path which can have tolerance,
hence the TolerancedNumber class.
"""

import numpy as np
from abc import ABC
from ..util import create_grid, rot_x, rot_y, rot_z
from ..util import plot_reference_frame
from ..samplers import Sampler, sample_SO3
from ..pyquat_extended import QuaternionExtended as Quaternion


class IsToleranced(ABC):
    is_toleranced: bool


class FixedNumber(IsToleranced):
    """
    Wrapper for a float to simplify `~acrobotics.path.path_pt.PathPt`.
    This class should never be used outside the path module.
    """

    def __init__(self, value: float):
        self.nominal = value
        self.is_toleranced = False

    def discretize(self):
        return self.nominal


class TolerancedNumber(FixedNumber, IsToleranced):
    """
    A range used to define path constraints.
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        nominal: float = None,
        num_samples: int = 10,
    ):
        self.upper = upper_bound
        self.lower = lower_bound
        self.num_samples = num_samples
        self.is_toleranced = True

        if nominal is None:
            self.nominal = lower_bound + (upper_bound - lower_bound) / 2
        else:
            self.nominal = nominal

    def discretize(self):
        return np.linspace(self.lower, self.upper, self.num_samples)

    def reduce_bounds(self, prev_value, reduction_factor):
        """Note: this function changes it's own state."""
        range_width = abs(self.upper - self.lower) / reduction_factor
        self.lower = max(prev_value - range_width / 2, self.lower)
        self.upper = min(prev_value + range_width / 2, self.upper)


class TolerancedQuaternion(IsToleranced):
    def __init__(
        self, quaternion: Quaternion, quat_distance: float, nominal_quaternion=None
    ):
        self.quat = quaternion
        self.dist = quat_distance
        self.nominal_quat = nominal_quaternion
        self.is_toleranced = True

    def discretize(self):
        raise NotImplementedError

    def reduce_bounds(self, prev_value, reduction_factor):
        self.quat = prev_value
        self.dist = self.dist / reduction_factor
