#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.path.toleranced_number import TolerancedNumber
from acrobotics.path.path_pt import *

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pyquaternion import Quaternion


class TestEulerPt:
    def test_pos_tol(self):
        x = TolerancedNumber(2, 4, num_samples=3)
        pos = [x, 2, 3]
        rpy = [4, 5, 6]
        tp = TolEulerPt(pos, rpy)
        samples = tp.sample_grid()

        actual_rot = [tf[:3, :3] for tf in samples]
        assert_almost_equal(actual_rot[0], actual_rot[1])
        assert_almost_equal(actual_rot[1], actual_rot[2])
        assert_almost_equal(actual_rot[2], actual_rot[0])

        actual_pos = np.array([tf[:3, 3] for tf in samples])
        desired_pos = np.array([[2, 2, 3], [3, 2, 3], [4, 2, 3]])
        assert_almost_equal(actual_pos, desired_pos)

        nominal_pos = tp.nominal_transform[:3, 3]
        assert_almost_equal(nominal_pos, np.array([3, 2, 3]))
        nominal_rot = tp.nominal_transform[:3, :3]
        assert_almost_equal(nominal_rot, actual_rot[0])

        actual_str = tp.__str__()
        assert actual_str == "[3. 2. 3.]"

        N = 10
        res = tp.sample_incremental(N, "random_uniform")
        assert len(res) == N
        assert res[0].shape == (4, 4)
        # res_desired = np.tile([1, 2, 3, 4, 5, 6], (N, 1))
        # # first column contains random samples soit is not compared
        # assert_almost_equal(res[:, 1:], res_desired[:, 1:])
        # assert max(res[:, 0]) <= 4.0
        # assert min(res[:, 0]) >= 2.0


class TestTolerancedNumber:
    # def test_nominal_outside_bounds_error(self):
    #     with pytest.raises(ValueError) as info:
    #         a = TolerancedNumber(0, 1, nominal=1.5)
    #     # check whether the error message is present
    #     msg = "ValueError: Nominal value must respect the bounds"
    #     print(info)
    #     assert msg in str(info)

    def test_get_initial_sampled_range(self):
        a = TolerancedNumber(0, 4, num_samples=5)
        a1 = a.discretize()
        d1 = [0, 1, 2, 3, 4]
        assert_almost_equal(a1, d1)
