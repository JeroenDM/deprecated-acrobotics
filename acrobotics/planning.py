#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for sampling based motion planning for path following.
"""
import numpy as np
from .cpp.graph import Graph
from .path import point_to_frame


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
        for pi in tp.discretise():
            q_sol = []
            Ti = point_to_frame(pi)
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


def cart_to_joint_no_redundancy(robot, path, scene):
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
        for Ti in tp.get_samples(1000, rep='transform'):
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
