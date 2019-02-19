#!/usr/bin/env python3
"""
A box open on the side.
A path inside this box with tolerance on orientation.
"""
import numpy as np
from ..geometry import Shape, Collection
from ..util import pose_x
from ..path import TolerancedNumber, TrajectoryPt

h = 1.0
w = 0.8
l = 1.6
t = 0.01

bottom = Shape(l, w, t)
top = Shape(l, w, t)
back = Shape(l-2*t, t, h)
left = Shape(t, w, h)
right = Shape(t, w, h)

#workpiece = [bottom, top, back, left, right]
workpiece_tf = [pose_x(0, 0, 0, t/2),
                pose_x(0, 0, 0, h + 3*t/2),
                pose_x(0, 0, w/2 - t/2, h/2 + t),
                pose_x(0, l/2 - t/2, 0, h/2 + t),
                pose_x(0, -l/2 + t/2, 0, h/2 + t)]

workpiece = Collection([bottom, top, back, left, right], workpiece_tf)

# welding torch path
N = 10
marge = 0.1
p_start = np.array([-l/2 + t + marge, w/2 - t, t])
p_goal  = np.array([ l/2 - t - marge, w/2 - t, t])

p_offset = np.array([0, -0.04, 0.04])

path = []
RX = -3*np.pi/4
tol_rx = TolerancedNumber(RX, RX-0.5, RX + 0.5, samples=3)
tol_ry = TolerancedNumber(0, -0.5, 0.5, samples=3)
tol_rz = TolerancedNumber(np.pi, 0, 2*np.pi, samples=10)

for par in np.linspace(0, 1, N):
    p_i = (1 - par) * p_start + par * p_goal + p_offset

    tp_i = TrajectoryPt([p_i[0], p_i[1], p_i[2], tol_rx, 0, tol_rz])
    path.append(tp_i)
