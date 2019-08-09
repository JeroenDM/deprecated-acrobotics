import numpy as np
import casadi as ca
from casadi import Opti, dot

from .geometry import Polyhedron


def create_collision_constraints(lam, mu, Ar, Ao, br, bo, eps=1e-6):
    cons = []
    cons.append(-dot(br, lam) - dot(bo, mu) >= eps)
    cons.append(Ar.T @ lam + Ao.T @ mu == 0.0)
    cons.append(dot(Ar.T @ lam, Ar.T @ lam) <= 1.0)
    cons.append(lam >= 0.0)
    cons.append(mu >= 0.0)
    return cons


def create_cc_for_joint_pose(robot, poly_robot, poly_scene, q, lamk, muk):
    """"
    NOTE: assumes only one shape / robot link
    """
    cons = []
    fk = robot.fk_all_links_casadi(q)
    for i in range(robot.ndof):
        Ri = fk[i][:3, :3]
        pi = fk[i][:3, 3]
        for j in range(len(poly_scene)):
            Ar = poly_robot[i].A @ Ri.T
            br = poly_robot[i].b + Ar @ pi
            cons.extend(
                create_collision_constraints(
                    lamk[i][j], muk[i][j], Ar, poly_scene[j].A, br, poly_scene[j].b
                )
            )
    return cons


def create_cc(opti, robot, scene, q):

    poly_robot = []
    for link in robot.links:
        poly_robot.extend(link.geometry.get_polyhedrons())
    poly_scene = scene.get_polyhedrons()

    S = len(poly_robot[0].b)
    nobs = len(poly_scene)
    N, ndof = q.shape
    # dual variables arranged in convenient lists to acces with indices
    lam = [
        [[opti.variable(S) for j in range(nobs)] for i in range(ndof)] for k in range(N)
    ]
    mu = [
        [[opti.variable(S) for j in range(nobs)] for i in range(ndof)] for k in range(N)
    ]

    cons = []
    for k in range(N):
        cons.extend(
            create_cc_for_joint_pose(
                robot, poly_robot, poly_scene, q[k, :], lam[k], mu[k]
            )
        )

    return cons
