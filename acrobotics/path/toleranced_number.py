"""
Tools to specify a path which can have tolerance,
hence the TolerancedNumber class.
"""

import numpy as np

from ..util import create_grid, rot_x, rot_y, rot_z
from ..util import plot_reference_frame
from ..samplers import Sampler, sample_SO3


class PathPointNumber:
    """
    Wrapper for a float to simplify `~acrobotics.path.path_pt.PathPt`.
    This class should never be used outside the path module.
    """

    def __init__(self, value: float):
        self.nominal = value
        self.is_toleranced = False

    def discretize(self):
        return self.nominal


class TolerancedNumber(PathPointNumber):
    """ A range used to define path constraints.
    """

    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        nominal: float = None,
        num_samples: int = 10,
    ):
        self.is_toleranced = True
        if nominal is None:
            self.nominal = lower_bound + (upper_bound - lower_bound) / 2
        elif (nominal < lower_bound) or (nominal > upper_bound):
            raise ValueError("Nominal value must respect the bounds.")
        else:
            self.nominal = nominal

        self.upper = upper_bound
        self.lower = lower_bound
        self.num_samples = num_samples

    def discretize(self):
        return np.linspace(self.lower, self.upper, self.num_samples)

