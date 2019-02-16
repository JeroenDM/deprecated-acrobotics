#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kuka forward kinematics.
Run this script and copy paste the result in the kuka_fk function.

@author: jeroen
"""

from numpy import cos, sin, array

def fk_kuka(q):
    q1, q2, q3, q4, q5, q6 = q
    a1=0.18; a2=0.6; d4=0.62; d6=0.115
    T = array([
[((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) - sin(q5)*sin(q2 + q3)*cos(q1))*cos(q6) + (sin(q1)*cos(q4) - sin(q4)*cos(q1)*cos(q2 + q3))*sin(q6), -((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) - sin(q5)*sin(q2 + q3)*cos(q1))*sin(q6) + (sin(q1)*cos(q4) - sin(q4)*cos(q1)*cos(q2 + q3))*cos(q6), (sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) + sin(q2 + q3)*cos(q1)*cos(q5), a1*cos(q1) + a2*cos(q1)*cos(q2) + d4*sin(q2 + q3)*cos(q1) + d6*((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) + sin(q2 + q3)*cos(q1)*cos(q5))],
[((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*cos(q5) - sin(q1)*sin(q5)*sin(q2 + q3))*cos(q6) - (sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*sin(q6), -((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*cos(q5) - sin(q1)*sin(q5)*sin(q2 + q3))*sin(q6) - (sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*cos(q6), (sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*sin(q5) + sin(q1)*sin(q2 + q3)*cos(q5), a1*sin(q1) + a2*sin(q1)*cos(q2) + d4*sin(q1)*sin(q2 + q3) + d6*((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*sin(q5) + sin(q1)*sin(q2 + q3)*cos(q5))],
[                                                                (sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q6) - sin(q4)*sin(q6)*sin(q2 + q3),                                                                 -(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q6) - sin(q4)*sin(q2 + q3)*cos(q6),                                     sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3),                                                                  a2*sin(q2) - d4*cos(q2 + q3) + d6*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))],
[                                                                                                                                                           0,                                                                                                                                                             0,                                                                                       0,                                                                                                                                                        1]])
    return T

if __name__ == "__main__":
    import sympy as sp
    from sympy.utilities.lambdify import lambdify
    from sympy import cos, sin
    
    def createDH(c, s, a, ca, sa, d):
        T1 = sp.Matrix([[c, -s, 0, 0],
                        [s,  c, 0, 0],
                        [0,  0, 1, d],
                        [0,  0, 0, 1]])
        T2 = sp.Matrix([[1, 0 , 0  , a],
                        [0, ca, -sa, 0],
                        [0, sa, ca , 0],
                        [0, 0 , 0  , 1]])
        return sp.trigsimp(T1 * T2)
    
    q1, q2, q3, q4, q5, q6, a1, a2, d4, d6 = sp.symbols('q1 q2 q3 q4 q5 q6 a1 a2 d4 d6')
    T1 = createDH(cos(q1), sin(q1), a1, 0, 1, 0)
    T2 = createDH(cos(q2), sin(q2), a2, 1, 0, 0)
    T3 = createDH(cos(q3), sin(q3), 0 , 0, 1, 0)
    T4 = createDH(cos(q4), sin(q4), 0 , 0, -1, d4)
    T5 = createDH(cos(q5), sin(q5), 0 , 0, 1, 0)
    T6 = createDH(cos(q6), sin(q6), 0 , 1, 0, d6)
    
    T = T1 * T2 * T3 * T4 * T5 * T6
    FK = sp.trigsimp(T)
    print(FK)

