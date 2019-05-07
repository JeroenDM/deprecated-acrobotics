from numpy import cos, sin, pi, sqrt
from numpy.random import uniform

class Sampler:
    """ Generate uniform random samples, also for quaternions.
    Inspered by ompl code organisation.
    """

    def __init__(self):
        pass


    def random_quaternion(self):
        """ http://planning.cs.uiuc.edu/node198.html
        """
        u1, u2, u3 = uniform(size=(3,))
        w = sqrt(1 - u1) * sin(2 * pi * u2)
        x = sqrt(1 - u1) * cos(2 * pi * u2)
        y = sqrt(u1) * sin(2 * pi * u3)
        z = sqrt(u1) * cos(2 * pi * u3)
        return Quaternion(x, y, z, w)

    def random_quaternion_near(self, q_near, distance):
        """ Sample around a given quaternion
        """
        if distance >= 0.25 * pi:
            return self.random_quaternion()
        else:
            d = uniform()
