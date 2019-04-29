import numpy as np
from acrobotics.path import TolerancedNumber, PathPt
# from pyquaterion import Quaterion

xt = TolerancedNumber(0.5, 1.5, samples=3)


tp1 = PathPt([xt, 2.0, 3.0])

print(tp1)
print(tp1.discretise())

tp2 = PathPt([1.0, 2.0, 3.0], [0.0, 0.0, xt])

print(tp2)
print(tp2.discretise())

tp3 = PathPt([1.0, 2.0, xt], [1.0, 0.0, 0.0, 0.0])

print(tp3)
print(tp3.discretise())
