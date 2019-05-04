#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from acrobotics.path import TolPositionPoint, TolerancedNumber
from acrobotics.path import FreeOrientationPt
from acrobotics.geometry import Shape, Collection
from pyquaternion import Quaternion


# sinusoid line with pos tolerance along x
# end-effector pointing down with fixed orientation
q1 = Quaternion(axis=[0, 1, 0], angle=np.pi)
dx = 0.05

path_pos_tol = []
for s in np.linspace(0, 1, 20):
    xi = 0.8 + 0.1 * np.sin(2 * np.pi * s)
    xt = TolerancedNumber(xi - dx, xi + dx)
    yi = s * 0.2 + (1-s) * (-0.2)
    zi = 0.2
    path_pos_tol.append(TolPositionPoint([xt, yi, zi], q1))

# straight line with free end-effector orientation
path_ori_free = []
for s in np.linspace(0, 1, 15):
    xi = 0.8
    yi = s * 0.2 + (1-s) * (-0.2)
    zi = 0.2
    path_ori_free.append(FreeOrientationPt([xi, yi, zi]))

# table with obstacle
table = Shape(0.5, 0.5, 0.1)
table_tf = np.array([[1, 0, 0, 0.80],
                    [0, 1, 0, 0.00],
                    [0, 0, 1, 0.12],
                    [0, 0, 0, 1]])

obstacle = Shape(0.1, 0.1, 0.5)
obstacle_tf = np.array([[1, 0, 0, 1.00],
                        [0, 1, 0, 0.25],
                        [0, 0, 1, 0.12],
                        [0, 0, 0, 1]])



scene = Collection([table, obstacle],
                   [table_tf, obstacle_tf])

table2 = Shape(0.5, 0.25, 0.1)
table2_tf = np.array([[1, 0, 0, 0.80],
                    [0, 1, 0, 0.25],
                    [0, 0, 1, 0.13],
                    [0, 0, 0, 1]])

box_up = Shape(0.1, 0.1, 0.1)
box_lo = Shape(0.1, 0.1, 0.1)
box_up_tf = np.array([[1, 0, 0, 0.80],
                    [0, 1, 0, -0.1],
                    [0, 0, 1, 0.32],
                    [0, 0, 0, 1]])
box_lo_tf = np.array([[1, 0, 0, 0.80],
                    [0, 1, 0, -0.1],
                    [0, 0, 1, 0.08],
                    [0, 0, 0, 1]])

scene2 = Collection([table2, box_up, box_lo],
                    [table2_tf, box_up_tf, box_lo_tf])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from acrobotics.util import get_default_axes3d,plot_reference_frame

    fig2, ax2 = get_default_axes3d([-1, 1], [-1, 1], [-1, 1])
    for pi in path_ori_free: pi.plot(ax2)
    scene.plot(ax2, c='g')
    plt.show()
