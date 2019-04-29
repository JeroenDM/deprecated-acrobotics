"""
Tools to specify a path which can have tolerance,
hence the TolerancedNumber class.
"""

import numpy as np
from pyquaternion import Quaternion
from .util import HaltonSampler, create_grid, rot_x, rot_y, rot_z, sample_SO3

# =============================================================================
# Classes
# =============================================================================


class TolerancedNumber:
    """ A range on the numner line used to define path constraints
    """

    def __init__(self, lower_bound, upper_bound, nominal=None, samples=10):

        if nominal is None:
            self.nominal = lower_bound + (upper_bound - lower_bound) / 2
        elif (nominal < lower_bound) or (nominal > upper_bound):
            raise ValueError("Nominal value must respect the bounds")
        else:
            self.nominal = nominal

        self.upper = upper_bound
        self.lower = lower_bound
        self.num_samples = samples

    def discretise(self):
        return np.linspace(self.lower, self.upper, self.num_samples)

class PathPt:
    """ Trajectory point for a desired end-effector pose in cartesian space

    saves orientation as quaternions, or rpy euler angles?
    returns 4 by 4 transform matrix if asked for discretisation
    """

    def __init__(self, *args, **kwargs):
        num_args  =len(args)

        # defaults
        self.rpy = None
        self.quat = None
        self.rpy_has_tol, self.nominal_rpy = [False]*3, np.zeros(3)
        self.quat_tol = None

        if num_args is 0:
            raise("Zero input arguments")

        elif num_args is 1:
            self.pos = args[0]
            self.pos_has_tol, self.nominal_pos = self._check_for_tolerance(self.pos)

        elif num_args is 2:
            self.pos = args[0]
            self.pos_has_tol, self.nominal_pos = self._check_for_tolerance(self.pos)

            if len(args[1]) is 3:
                self.rpy = args[1]
                self.rpy_has_tol, self.nominal_rpy = self._check_for_tolerance(self.rpy)

            elif len(args[1]) is 4:
                self.quat = args[1]
                self.nominal_rpy, self.rpy_has_tol = None, None
                if "quat_tol" in kwargs:
                    self.quat_tol = kwargs["quat_tol"]

            else:
                raise ValueError("Second argument needs a length of 3 or 4.")

        else:
            raise ValueError("Too much input arguments.")

        self.timing = 0.1  # with respect to previous point

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
        if self.rpy is not None:
            ori = self.nominal_rpy
        elif self.quat is not None:
            ori = self.quat
        else:
            ori = 'Free'

        return  'position: {}\norientation: {}'.format(
                    self.nominal_pos,
                    ori
                )

    def discretise(self):
        """ Returns a discrete version of the range of a trajectory point
        """
        r = []
        # discretise position
        for i in range(3):
            if self.pos_has_tol[i]:
                r.append(self.pos[i].discretise())
            else:
                r.append(self.pos[i])

        # discretise roll pitch yaw angles
        if self.rpy is not None:
            for i in range(3):
                if self.rpy_has_tol[i]:
                    r.append(self.rpy[i].discretise())
                else:
                    r.append(self.rpy[i])

        # discretise quaterion
        if self.quat is not None:
            if self.quat_tol is None:
                r.extend(self.quat)
            else:
                pass

        return create_grid(r)

class TolPositionPoint:
    """ Path point with fixed orientation and tol on position
    """

    def __init__(self, pos, quat):
        self.pos = pos
        self.pos_has_tol, self.pos_nom = self._check_for_tolerance(pos)
        self.quat = quat

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

    def get_samples(self, samples, rep=None, dist=None):
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

    def plot(self, ax):
        ax.plot([self.pos_nom[0]], [self.pos_nom[1]], [self.pos_nom[2]], 'o', c='r')


class FreeOrientationPt:
    """ Trajectory point with fixed position and free orientation.
    Work in progress
    """

    def __init__(self, position):
        self.p = np.array(position)

    def get_samples(self, num_samples, rep='rpy', dist=None):
        """ Sample orientation, position is fixed for every sample
        """
        if rep == 'rpy':
            rpy = np.array(sample_SO3(n=num_samples, rep='rpy'))
            pos = np.tile(self.p, (num_samples, 1))
            return np.hstack((pos, rpy))
        if rep == 'transform':
            Ts = np.array(sample_SO3(n=num_samples, rep='transform'))
            for Ti in Ts:
                Ti[:3, 3] = self.p
            return Ts

    def discretise(self):
        return self.get_samples(100)

    def __str__(self):
        return str(self.p)

    def plot(self, ax):
        ax.plot([self.p[0]], [self.p[1]], [self.p[2]], 'o', c='r')

class TolOrientationPt:
    def __init__(self, position, orientation):
        self.p = np.array(position)
        self.o = orientation

    def get_samples(self, num_samples, rep='rpy', dist=0.1):
        """ Sample orientation, position is fixed for every sample
        """
        if rep == 'rpy':
            rpy = np.array(sample_SO3(n=num_samples, rep='rpy'))
            pos = np.tile(self.p, (num_samples, 1))
            return np.hstack((pos, rpy))
        if rep == 'transform':
            Ts = []
            print('sampling near with distance dist')
            for i in range(num_samples):
                qr = Quaternion.random_near(self.o, dist)
                Ts.append(qr.transformation_matrix)
            Ts = np.array(Ts)
            for Ti in Ts:
                Ti[:3, 3] = self.p
            return Ts

    def discretise(self):
        return self.get_samples(100)

    def __str__(self):
        return str(self.p)

    def plot(self, ax):
        ax.plot([self.p[0]], [self.p[1]], [self.p[2]], 'o', c='r')

# =============================================================================
# Functions
# =============================================================================


def point_to_frame(p):
    """ Convert pose as 6-element vector to transform matrix

    [x, y, z, ox, oy, oz] three position parameters and three paramters
    for the xyz-euler angles.
    """
    p = np.array(p)
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = np.dot(rot_x(p[3]), np.dot(rot_y(p[4]), rot_z(p[5])))
    return T
