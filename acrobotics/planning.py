#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for sampling based motion planning for path following.
"""
import numpy as np
from .cpp.graph import Graph
from .path import *
from pyquaternion import Quaternion

DEBUG = False

def cart_to_joint_simple(robot, path, scene, q_fixed):
    """ cartesian path to joint solutions

    q_fixed argument provides an already sampled version
    for the redundant joints.
    The path tolerance is sampled by tp.discretise inside
    the TrajectoryPoint class.

    Return array with float32 elements, reducing data size.
    During graph search c++ floats are used, also float32.
    """
    Q = []
    for i, tp in enumerate(path):
        print('Processing point ' + str(i) + '/' + str(len(path)))
        for Ti in tp.discretise():
            q_sol = []
            for qfi in q_fixed:
                sol = robot.ik(Ti, qfi)
                if sol['success']:
                    for qi in sol['sol']:
                        if not robot.is_in_collision(qi, scene):
                            q_sol.append(qi)
        if len(q_sol) > 0:
            Q.append(np.vstack(q_sol).astype('float32'))
        else:
            Q.append([])
    return Q

class PathPtType:
    TOL_POS = 0
    TOL_ORI = 1
    AXIS = 2

def get_new_bounds(l, u, m, red=4):
    """ TODO reduction of 2 does not reduce interval
    """
    delta = abs(u - l) / red
    l_new = max(m - delta, l)
    u_new = min(m + delta, u)
    return l_new, u_new

def resample_trajectory_point(tp, pos_ee, quat):
    """ create a new trajectory point with smaller bounds
    """
    p_new = []
    for i, val in enumerate(tp.pos):
        if tp.pos_has_tol[i]:
            # check for rounding errors on pfk
            # TODO move this code into toleranced number
            if pos_ee[i] < val.lower:
                pos_ee[i] = val.lower
            if pos_ee[i] > val.upper:
                pos_ee[i] = val.upper
            # now calculate new bounds with corrected pkf
            l, u = get_new_bounds(val.lower, val.upper, pos_ee[i])
            val_new = TolerancedNumber(l, u, samples=val.num_samples)
        else:
            val_new = val
        p_new.append(val_new)
    return TolPositionPoint(p_new, quat)

class SolutionPoint:
    """ class to save intermediate solution info for trajectory point
    """
    def __init__(self, tp):
        self.tp_init = tp
        self.tp_current = tp
        self.q_best = None
        self.jl = []

        self.tol_type = None
        if isinstance(tp, FreeOrientationPt):
            self.tol_type = PathPtType.TOL_ORI
        elif isinstance(tp, TolPositionPoint):
            self.tol_type = PathPtType.TOL_POS
        elif isinstance(tp, AxisAnglePt):
            self.tol_type = PathPtType.AXIS
        else:
            raise ValueError("Unkown path point type")

        self.samples = None
        self.joint_solutions = np.array([])
        self.num_js = 0
        # distance of pi / 4 is almost the same as no tolerance
        self.quat_dist_tol = np.pi / 4


    def calc_joint_solutions(self, robot, tp_samples, check_collision=False, scene=None):
        """ Convert a cartesian trajectory point to joint space """
        # input validation
        if check_collision:
            if scene == None:
                raise ValueError("scene is needed for collision checking")

        #tp_discrete = self.tp_current.discretise()
        joint_solutions = []
        for Ti in tp_samples:
            sol = robot.ik(Ti)
            if sol['success']:
                for qsol in sol['sol']:
                    if check_collision:
                        if not robot.is_in_collision(qsol, scene):
                            joint_solutions.append(qsol)
                    else:
                        joint_solutions.append(qsol)

        return np.array(joint_solutions)

    def resample(self, robot, reduction=2):
        """ Create a toleranced path around the current solution.
        Reduce the tolerance range by a factor 'reduction'.
        """

        # calculate forward kinematics for the current solution
        Tee = robot.fk(self.q_best)
        pos = [Tee[0, 3], Tee[1, 3], Tee[2, 3]]
        qee = Quaternion(matrix=Tee)

        if self.tol_type is PathPtType.TOL_ORI:
            self.tp_current = TolOrientationPt(pos, qee)
            self.quat_dist_tol = self.quat_dist_tol / reduction
        elif self.tol_type is PathPtType.TOL_POS:
            ## create reduced tolerance point
            self.tp_current = resample_trajectory_point(self.tp_current, pos, qee)
        elif self.tol_type is PathPtType.AXIS:
            tp = self.tp_current
            # assume symmetric bounds around current rotation axis
            new_tol_angle = TolerancedNumber(
                tp.angle.lower / 2,
                tp.angle.upper / 2,
                samples=tp.angle.num_samples
            )
            self.tp_current = AxisAnglePt(pos, qee.axis, new_tol_angle, qee)
        else:
            raise ValueError("tolerance type not set.")

    def get_samples(self, num_samples):
        return self.tp_current.get_samples(num_samples,
            rep='transform',
            dist=self.quat_dist_tol)

def cart_to_joint_iterative(robot, path, scene, num_samples=1000, max_iters=3):
    """ cartesian path to joint solutions

    The path tolerance is sampled by tp.discretise inside
    the TrajectoryPoint class.

    Return array with float32 elements, reducing data size.
    During graph search c++ floats are used, also float32.
    """
    current_path = [SolutionPoint(tp) for tp in path]
    costs = []

    for iter in range(max_iters):
        Q = []
        for i, tp in enumerate(current_path):
            print('Processing point ' + str(i) + '/' + str(len(path)))
            samples = tp.get_samples(num_samples)
            Q.append(tp.calc_joint_solutions(robot, samples,
                check_collision=True,
                scene=scene).astype('float32'))
        if np.all([len(qi) for qi in Q]):
            print('Found collision free configurations for every tp.')
            sol = get_shortest_path(Q, method='dijkstra')
            if sol['success']:
                costs.append(sol['length'])
                for qi, tpi in zip(sol['path'], current_path):
                    tpi.q_best = qi
                    tpi.resample(robot)
            else:
                print('failed to find shortest path in graph at iter: {}'.format(iter))
                return {'success': False}
        else:
            print('Not every tp has collision free configurations.')
            return {'success': False}

    sol['costs'] = costs
    return sol

def cart_to_joint_no_redundancy(robot, path, scene, num_samples=1000):
    """ cartesian path to joint solutions

    The path tolerance is sampled by tp.discretise inside
    the TrajectoryPoint class.

    Return array with float32 elements, reducing data size.
    During graph search c++ floats are used, also float32.
    """
    Q = []
    for i, tp in enumerate(path):
        print('Processing point ' + str(i) + '/' + str(len(path)))
        q_sol = []
        for Ti in tp.get_samples(num_samples, rep='transform'):
            sol = robot.ik(Ti)
            if sol['success']:
                for qi in sol['sol']:
                    if not robot.is_in_collision(qi, scene):
                        q_sol.append(qi)
                    else:
                        if DEBUG:
                            print("Collision for point: {}".format(Ti[:3, 3]))
            else:
                if DEBUG:
                    print("IK failed for point: {}".format(Ti[:3, 3]))

        if len(q_sol) > 0:
            Q.append(np.vstack(q_sol).astype('float32'))
        else:
            Q.append([])
    return Q


def cart_to_joint_tool_first_cc(robot, path, scene):
    """ cartesian path to joint solutions

    In this vesion the tool is first checked for collision for every
    sample before solving the inverse kinematics and checking for
    collision with the rest of the robot.
    """
    # no tool then use last link as a tool for collision checking
    shape_last_link = robot.links[-1].shape
    tf_last_link = robot.links[-1].tf_shape
    tf_move = np.eye(4)
    Q = []
    for i, tp in enumerate(path):
        print('Processing point ' + str(i) + '/' + str(len(path)))
        q_sol = []
        for Ti in tp.get_samples(1000, rep='transform'):
            # move last link to Ti and check for collision
            tf_move = np.dot(tf_last_link, Ti)
            shape_last_link.set_transform(tf_move)
            if shape_last_link.is_in_collision_multi(scene.get_shapes()):
               continue
            sol = robot.ik(Ti)
            if sol['success']:
                for qi in sol['sol']:
                    if not robot.is_in_collision(qi, scene):
                        q_sol.append(qi)

        if len(q_sol) > 0:
            Q.append(np.vstack(q_sol).astype('float32'))
        else:
            Q.append([])
    return Q

def get_shortest_path(Q, method='bfs'):
    """ Calculate the shortest path from joint space data

    When the path with trajectory points is converted to joint space,
    this data can be used to construct a graph and look for the shortest path.
    The current distance metrix is the l1-norm of joint position difference
    between two points.

    I still have to implement maximum joint movement and acceleration limits.
    So technically this will always find a shortest path for now.

    Parameters
    ----------
    Q : list of nympy.ndarrays of float
        A list with the possible joint positions for every trajectory point
        along a path.

    Returns
    -------
    dict
        A dictionary with a key 'success' to indicate whether a path was found.
        If success is True, then the key 'path' contains a list with the joint
        position for every trajectory point that gives the shortest path.

    Notes
    -----
    I have a problem with swig type conversions. Therefore the type of the
    input data is checked and changed from float64 to float32.
    """
    Q = _check_dtype(Q)

    n_path = len(Q)
    # initialize graph
    g = Graph()
    for c in Q:
        if len(c) == 0:
            # one of the trajectory points is not reachable
            return {'success': False}
        g.add_data_column(c)
    g.init()

    # run shortest path algorithm
    if method == 'bfs':
        g.run_bfs()
    elif method == 'dijkstra':
        g.run_dijkstra()
    else:
        raise NotImplementedError(
            'The method {} is not implented yet.'.format(method))
    # print result
    # g.print_graph()
    g.print_path()

    # get joint values for the shortest path
    p_i = g.get_path(n_path)
    cost = g.get_path_cost()

    if p_i[0] == -1:
        return {'success': False}
    else:
        res = []
        for k, i in zip(range(n_path), p_i):
            # TODO ugly all the "unsave" typecasting
            qki = Q[k][i].astype('float64')
            res.append(qki)

        return {'success': True, 'path': res, 'length': cost}


def _check_dtype(Q):
    """ Change type if necessary to float32

    Due to an unresolved issue with swig and numpy, I have to convert the type.

    Parameters
    ----------
    Q : list of nympy.ndarrays of float
        A list with the possible joint positions for every trajectory point
        along a path.

    Returns
    -------
    list of nympy.ndarrays of float32
    """
    if Q[0].dtype != 'float32':
        print("converting type of Q")
        for i in range(len(Q)):
            Q[i] = Q[i].astype('float32')

    return Q
