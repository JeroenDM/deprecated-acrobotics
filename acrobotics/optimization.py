import numpy as np
import casadi as ca
from casadi import cos, sin, dot

def get_link_relative_transform(link, qi):
    """ Link transform according to the Denavit-Hartenberg convention.
    Casadi compatible function.
    """
    a, alpha, d, theta = link.dh.a, link.dh.alpha, link.dh.d, qi

    c_t, s_t = cos(theta), sin(theta)
    c_a, s_a = cos(alpha), sin(alpha)

    row1 = ca.hcat([c_t, -s_t * c_a,  s_t * s_a,  a * c_t])
    row2 = ca.hcat([s_t,  c_t * c_a, -c_t * s_a,  a * s_t])
    row3 = ca.hcat([0, s_a, c_a, d])
    row4 = ca.hcat([0, 0, 0, 1])

    return ca.vcat([row1, row2, row3, row4])

def fk_all_links(links, q, tf_base=np.eye(4)):
    """ Return link frames (not base or tool).
    Casadi compatible function.
    """
    tf_links = []
    T = tf_base
    for i in range(len(links)):
        Ti = get_link_relative_transform(links[i], q[i])
        T = T @ Ti
        tf_links.append(T)
    return tf_links

def fk_kuka2(q):
    """ Forward kinematics for a fixed version of the Kuka kr5 arc.
    Casadi compatible function.
    """
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2], q[3], q[4], q[5]
    a1=0.18; a2=0.6; d4=0.62; d6=0.115
    T = np.array([
[((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) - sin(q5)*sin(q2 + q3)*cos(q1))*cos(q6) + (sin(q1)*cos(q4) - sin(q4)*cos(q1)*cos(q2 + q3))*sin(q6), -((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*cos(q5) - sin(q5)*sin(q2 + q3)*cos(q1))*sin(q6) + (sin(q1)*cos(q4) - sin(q4)*cos(q1)*cos(q2 + q3))*cos(q6), (sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) + sin(q2 + q3)*cos(q1)*cos(q5), a1*cos(q1) + a2*cos(q1)*cos(q2) + d4*sin(q2 + q3)*cos(q1) + d6*((sin(q1)*sin(q4) + cos(q1)*cos(q4)*cos(q2 + q3))*sin(q5) + sin(q2 + q3)*cos(q1)*cos(q5))],
[((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*cos(q5) - sin(q1)*sin(q5)*sin(q2 + q3))*cos(q6) - (sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*sin(q6), -((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*cos(q5) - sin(q1)*sin(q5)*sin(q2 + q3))*sin(q6) - (sin(q1)*sin(q4)*cos(q2 + q3) + cos(q1)*cos(q4))*cos(q6), (sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*sin(q5) + sin(q1)*sin(q2 + q3)*cos(q5), a1*sin(q1) + a2*sin(q1)*cos(q2) + d4*sin(q1)*sin(q2 + q3) + d6*((sin(q1)*cos(q4)*cos(q2 + q3) - sin(q4)*cos(q1))*sin(q5) + sin(q1)*sin(q2 + q3)*cos(q5))],
[                                                                (sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*cos(q6) - sin(q4)*sin(q6)*sin(q2 + q3),                                                                 -(sin(q5)*cos(q2 + q3) + sin(q2 + q3)*cos(q4)*cos(q5))*sin(q6) - sin(q4)*sin(q2 + q3)*cos(q6),                                     sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3),                                                                  a2*sin(q2) - d4*cos(q2 + q3) + d6*(sin(q5)*sin(q2 + q3)*cos(q4) - cos(q5)*cos(q2 + q3))],
[                                                                                                                                                           0,                                                                                                                                                             0,                                                                                       0,                                                                                                                                                        1]])
    return T
