import numpy as np
import casadi as ca
from casadi import cos, sin, dot
from acrobotics.util import get_default_axes3d
from acrobotics.recources.path_on_table import scene2
from acrobotics.recources.robots import Kuka
from acrobotics.optimization import fk_all_links, fk_kuka2

robot = Kuka()

scene = scene2

N = 10 # path discretization
# number of planes in the polyhedrons (the same for all shapes, robot and obstacles!)
S = 6
ndof = robot.ndof  #  robot degrees of freedom
nobs = len(scene.shapes)  #  number of obstacles
eps = 1e-6 # collision tolerance

# TASK: the end-effector path (no orientation)
xp = np.ones(N) * 0.8
yp = np.linspace(-0.2, 0.2, N)
zp = np.ones(N) * 0.2

pol_mat_a = []
pol_mat_b = []

# assume all links have only one shape
# not including forward kinematics pose
for link in robot.links:
    s = link.geometry.shapes[0]
    Ai, bi = s.get_polyhedron(np.eye(4))
    pol_mat_a.append(Ai)
    pol_mat_b.append(bi)

pol_mat_a_scene , pol_mat_b_scene = scene.get_polyhedron()

opti = ca.Opti()

q = opti.variable(N, 6)  #  joint variables along path
# dual variables arranged in convenient lists to acces with indices
lam = [[[opti.variable(S) for j in range(nobs)] for i in range(ndof)] for k in range(N)]
mu =  [[[opti.variable(S) for j in range(nobs)] for i in range(ndof)] for k in range(N)]


def col_con(lam, mu, Ar, Ao, br, bo):
    opti.subject_to( -dot(br, lam) - dot(bo, mu) >= eps)
    opti.subject_to(        Ar.T @ lam + Ao.T @ mu == 0.0)
    opti.subject_to( dot(Ar.T @ lam, Ar.T @ lam) <= 1.0)
    opti.subject_to(                         lam >= 0.0)
    opti.subject_to(                          mu >= 0.0)


for k in range(N):
    fk = fk_all_links(robot.links, q[k, :])
    for i in range(ndof):
        Ri = fk[i][:3, :3]
        pi = fk[i][:3, 3]
        for j in range(nobs):
            Ar = pol_mat_a[i] @ Ri.T
            br = pol_mat_b[i] + Ar @ pi
            col_con(lam[k][i][j], mu[k][i][j],
                    Ar,
                    pol_mat_a_scene[j],
                    br,
                    pol_mat_b_scene[j])

V = ca.sum1( (ca.sum2(q[:-1, :] - q[1:, :])**2 ))# + 0.05* ca.sumsqr(q) #+ 1 / ca.sum1(q[:, 4]**2)

opti.minimize(V)

for i in range(N):
    Ti = fk_kuka2(q[i, :])
    opti.subject_to(xp[i] == Ti[0, 3])
    opti.subject_to(yp[i] == Ti[1, 3])
    opti.subject_to(zp[i] == Ti[2, 3])

opti.solver('ipopt')

sol = opti.solve()

qp_sol = opti.value(q)

import matplotlib.pyplot as plt
fig, ax = get_default_axes3d([0, 1], [-0.5, 0.5], [0, 1])
ax.scatter(xp, yp, zp)
scene.plot(ax, c='g')
robot.animate_path(fig, ax, qp_sol)
ax.set_axis_off()
ax.view_init(24, 50)
plt.show()
