"""
Tools to specify a path which can have tolerance, hence the TolerancedNumber class.

@author: jeroen
"""

import numpy as np
from .util import HaltonSampler, create_grid, rot_x, rot_y, rot_z, sample_SO3
from numpy import sin, cos, sqrt, pi

#=============================================================================
# Classes
#=============================================================================

class TolerancedNumber:
    """ A range on the numner line used to define path constraints
    
    It also has a nominal value in the range which can be used in cost
    functions of some sort in the future. For example, if it is preffered
    that the robot end-effector stays close to an optimal pose, but can
    deviate if needed to avoid collision.
    
    Attributes
    ----------
    n : float
        Nominal / preffered value for this number
    u : float
        Upper limit.
    l : float
        lower limit
    s : int
        Number of samples used to produce descrete version of this number.
    range : numpy.ndarray of float
        A sampled version of the range on the number line, including limits.
        The nominal value is not necessary included.
    
    Notes
    -----
    Sampling for orientation is done uniformly at the moment.
    In 3D this is no good and special sampling techniques for angles should be
    used.
    The sampled range is now an attribute, but memory wise it would be better
    if it is a method. Then it is only descritized when needed.
    But it the number of path points will probably be limited so I preffer this
    simpler implementation for now.
    """
    def __init__(self, nominal, lower_bound, upper_bound, samples=10):
        """ Create a toleranced number
        
        Nominal does not have to be in the middle,
        it is the preffered value when we ever calculate some kind of cost.
        
        Parameters
        ----------
        nominal : float
            Nominal / preffered value for this number
        lower_bound : float
        upper_bound : float
        samples : int
            The number of samples taken when a sampled version of the number
            is returned in the range attribute. (default = 10)
        
        Examples
        --------
        >>> a = TolerancedNumber(0.7, 0.0, 1.0, samples = 6)
        >>> a.range
        array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
        """
        if nominal < lower_bound or nominal > upper_bound:
            raise ValueError("nominal value must respect the bounds")
        self.n = nominal
        self.u = upper_bound
        self.l = lower_bound
        self.s = samples
        self.range = np.linspace(self.l, self.u, self.s)
    
    def set_samples(self, samples):
        self.s = samples
        self.range = np.linspace(self.l, self.u, self.s)

class TrajectoryPt:
    """ Trajectory point for a desired end-effector pose in cartesian space
    
    This object bundles the constraints on the end-effector for one point
    of a path.
    
    Attributes
    ----------
    dim : int
        Pose dimensions, 3 for 2D planning, 6 for 3D planning.
    p : list of numpy.ndarray of float or ppr.path.TolerancedNumber
        Pose constraints for the end-effector (x, y, orientation).
        Can be a fixed number (a float) or a TolerancedNumber
    hasTolerance : list of bool
        Indicates which elements of the pose are toleranced (True) and
        fixed (False).
    p_nominal : list of float
        Same as p for fixed poses, the nominal value for a TolerancedNumber.
    timing : float
        Time in seconds it should be executed relative to the previous path
        point. Not used in current version.
    
    Examples
    --------
    Create a trajectory point at position (1.5, 3.1) with a symmetric
    tolerance of 0.4 on the x position.
    The robot orientation should be between 0 and pi / 4.
    (No preffered orientation, so assumed in the middle, pi / 8)

    >>> x = TolerancedNumber(1.5, 1.0, 2.0)
    >>> y = 3.1
    >>> angle = TolerancedNumber(np.pi / 8, 0.0, np.pi / 4)
    >>> tp = TrajectoryPt([x, y, angle])
    >>> tp.p_nominal
    array([ 1.5       ,  3.1       ,  0.39269908])
    
    A path is created by putting several trajectory points in a list.
    For example a vertical path with tolerance along the x-axis:
    
    >>> path = []
    >>> path.append(TrajectoryPt([TolerancedNumber(1.5, 1.0, 2.0), 0.0, 0]))
    >>> path.append(TrajectoryPt([TolerancedNumber(1.5, 1.0, 2.0), 0.5, 0]))
    >>> path.append(TrajectoryPt([TolerancedNumber(1.5, 1.0, 2.0), 1.0, 0]))
    >>> for p in path: print(p)
    [ 1.5  0.   0. ]
    [ 1.5  0.5  0. ]
    [ 1.5  1.   0. ]
    """
    def __init__(self, pee):
        """ Create a trajectory point from a given pose
        
        [x_position, y_position, angle last joint with x axis]
        
        Parameters
        ----------
        pee : list or numpy.ndarray of float or ppr.path.TolerancedNumber
            Desired pose of the end-effector for this path point,
            every value can be either a float or a TolerancedNumber
        """
        self.dim = len(pee)
        self.p = pee
        self.hasTolerance = [isinstance(pee[i], TolerancedNumber) for i in range(self.dim)]
        p_nominal = []
        for i in range(self.dim):
            if self.hasTolerance[i]:
                p_nominal.append(self.p[i].n)
            else:
                p_nominal.append(self.p[i])
        self.p_nominal = np.array(p_nominal)
        self.timing = 0.1 # with respect to previous point
        
        # for use of halton sampling
        # dimension is the number of toleranced numbers
        self.hs = HaltonSampler(sum(self.hasTolerance))
    
    def __str__(self):
        """ Returns string representation for printing
        
        Returns
        -------
        string
            List with nominal values for x, y and orientation.
        """
        return str(self.p_nominal)
    
    def discretise(self):
        """ Returns a discrete version of the range of a trajectory point
        
        Based on the sampled range in the Toleranced Numbers, a 3 dimensional grid
        representing end-effector poses that obey the trajectory point constraints.
        
        Parameters
        ----------
        pt : ppr.path.TrajectoryPt
        
        Returns
        -------
        numpy.ndarray
            Array with shape (M, 3) containing M possible poses for the robot
            end-effector that  obey the trajectory point constraints.
        
        Examples
        --------
        >>> x = TolerancedNumber(1, 0.5, 1.5, samples=3)
        >>> y = TolerancedNumber(0, -1, 1, samples=2)
        >>> pt = TrajectoryPt([x, y, 0])
        >>> pt.discretise()
        array([[ 0.5, -1. ,  0. ],
               [ 1. , -1. ,  0. ],
               [ 1.5, -1. ,  0. ],
               [ 0.5,  1. ,  0. ],
               [ 1. ,  1. ,  0. ],
               [ 1.5,  1. ,  0. ]])
        """
        r = []
        for i in range(self.dim):
            if self.hasTolerance[i]:
                r.append(self.p[i].range)
            else:
                r.append(self.p[i])
        grid = create_grid(r)
        return grid
    
    def get_samples(self, n, method='random'):
        """ Return sampled trajectory point based on values between 0 and 1
        """
        # check input
        sample_dim = sum(self.hasTolerance) # count the number of toleranced numbers
        
        if method == 'random':
            r = np.random.rand(n, sample_dim)
        elif method == 'halton':
            r = self.hs.get_samples(n)
        else:
            raise ValueError("Method not implemented.")
            
        # arrange in array and rescale the samples
        samples = []
        cnt = 0
        for i, val in enumerate(self.p):
            if self.hasTolerance[i]:
                samples.append(r[:, cnt] * (val.u - val.l) + val.l)
                cnt += 1
            else:
                samples.append(np.ones(n) * val)
        
        return np.vstack(samples).T

class FreeOrientationPt:
    """ Work in progress """
    def __init__(self, position):
        self.p = np.array(position)
    
    def get_samples(self, num_samples, rep='rpy'):
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
  
#=============================================================================
# Functions
#=============================================================================
def point_to_frame(p):
    """ Convert pose as 6-element vector to transform matrix
    
    [x, y, z, ox, oy, oz] three position parameters and three paramters
    for the xyz-euler angles.
    """
    p = np.array(p)
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = np.dot(rot_x(p[3]) , np.dot(rot_y(p[4]), rot_z(p[5])))
    return T