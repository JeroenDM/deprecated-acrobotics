import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from acrobotics.dp import *


def test_calculate_value_function():
    C1 = np.array([550, 900, 770], ndmin=2)
    C2 = np.array([[680, 790, 1050], [580, 760, 660], [510, 700, 830]])
    C3 = np.array([[610, 790], [540, 940], [790, 270]])
    C4 = np.array([[1030], [1390]])
    transition_costs = [C1, C2, C3, C4]

    vind, v = calculate_value_function(transition_costs)

    v_exact = [
        np.array([2870.0]),
        np.array([2320, 2220, 2150]),
        np.array([1640, 1570, 1660]),
        np.array([1030, 1390]),
        np.array([0.0]),
    ]

    v_ind_exact = [[0], [0, 0, 0], [0, 0, 1], [0, 0], [0]]

    print(v)
    assert len(v) == len(v_exact)
    for v1, v2 in zip(v, v_exact):
        assert_almost_equal(v1, v2)

    assert len(vind) == len(v_ind_exact)
    for v1, v2 in zip(vind, v_ind_exact):
        assert_almost_equal(v1, v2)


@pytest.fixture
def simple_graph():
    """ Define a simple graph with three stages and (2, 3, 3) states in
    those stages. Also define exact solutions for transition_costs,
    value function and indices to retreive shortest path, and the
    shortest path itself.
    """
    data1 = np.array([[0, 0], [0, 1]])
    data2 = np.array([[1, -1], [1, 0], [1, 1]])
    data3 = np.array([[0, 2], [2, 2]])
    data = [data1, data2, data3]
    C_exact = [np.array([[2, 1, 2], [3, 2, 1]]), np.array([[4, 4], [3, 3], [2, 2]])]
    v_exact = [np.array([4, 3]), np.array([4, 3, 2]), np.array([0, 0])]
    v_ind_exact = [[1, 2], [0, 0, 0], [0, 0]]
    path = [[0, 1], [1, 1], [0, 2]]
    return data, C_exact, v_exact, v_ind_exact, path


def test_apply_cost_function(simple_graph):
    def f(d1, d2):
        """L1 norm cost function. """
        ci = np.zeros((len(d1), len(d2)))
        for i in range(len(d1)):
            for j in range(len(d2)):
                ci[i, j] = np.sum(np.abs(d1[i] - d2[j]))
        return ci

    C = apply_cost_function(simple_graph[0], f)

    assert len(C) == len(simple_graph[1])
    for ca, cb in zip(C, simple_graph[1]):
        assert ca.shape == cb.shape
        assert_almost_equal(ca, cb)


def test_calculate_value_function(simple_graph):
    v_ind, v = calculate_value_function(simple_graph[1])

    print(v)
    assert len(v) == len(simple_graph[2])
    for v1, v2 in zip(v, simple_graph[2]):
        assert_almost_equal(v1, v2)

    assert len(v_ind) == len(simple_graph[3])
    for v1, v2 in zip(v_ind, simple_graph[3]):
        assert_almost_equal(v1, v2)


def test_extract_shortest_path(simple_graph):
    shortest_path = extract_shortest_path(
        simple_graph[0], simple_graph[3], simple_graph[2]
    )

    assert len(shortest_path) == len(simple_graph[4])
    for v1, v2 in zip(shortest_path, simple_graph[4]):
        assert_almost_equal(v1, v2)
