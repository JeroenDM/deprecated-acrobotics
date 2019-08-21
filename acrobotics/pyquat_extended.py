import numpy as np
from pyquaternion import Quaternion


class QuaternionExtended(Quaternion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def random_near(self, distance):
        """ Generate random unit quaternion near a given quaternion.
      Based on the 'SO3StateSpace' implementation in the open motion
      planning library (http://ompl.kavrakilab.org/).
      Params:
          other: a Quaternion around wich to sample
          distance: the maximum distance from the given Quaternion
      """

        if distance > 0.25 * np.pi:
            return super().random()

        d = np.random.random()
        g1, g2, g3 = np.random.normal(size=(3,))
        q_delta = Quaternion(axis=[g1, g2, g3], angle=(2 * d ** (1 / 3) * distance))

        return self * q_delta

