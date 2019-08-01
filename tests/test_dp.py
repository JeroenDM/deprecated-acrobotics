import numpy as np
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

    v_ind_exact = [[0], [0, 0, 0], [0, 0, 0], [0, 0], [0]]

    print(v)
    assert len(v) == len(v_exact)
    for v1, v2 in zip(v, v_exact):
        assert_almost_equal(v1, v2)

    assert len(vind) == len(v_ind_exact)
    for v1, v2 in zip(vind, v_ind_exact):
        assert_almost_equal(v1, v2)
