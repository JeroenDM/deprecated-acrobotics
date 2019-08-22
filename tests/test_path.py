#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.path.toleranced_number import TolerancedNumber, PathPointNumber
from acrobotics.path.path_pt import TolEulerPt, FreeOrientationPt
from acrobotics.samplers import SampleMethod

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

        # There's no tolerance on the rpy values,
        # so all rotation matrices should be equal
        actual_rot = [tf[:3, :3] for tf in samples]
        assert_almost_equal(actual_rot[0], actual_rot[1])
        assert_almost_equal(actual_rot[1], actual_rot[2])
        assert_almost_equal(actual_rot[2], actual_rot[0])

        # The x value samples must be [2, 3, 4]
        # the y and z value are fixed
        actual_pos = np.array([tf[:3, 3] for tf in samples])
        desired_pos = np.array([[2, 2, 3], [3, 2, 3], [4, 2, 3]])
        assert_almost_equal(actual_pos, desired_pos)

        # nominal x value in the middle of the range [2, 4] -> 3
        nominal_pos = tp.nominal_transform[:3, 3]
        assert_almost_equal(nominal_pos, np.array([3, 2, 3]))

        # nominal rotation matrix must equal the sampled one
        nominal_rot = tp.nominal_transform[:3, :3]
        assert_almost_equal(nominal_rot, actual_rot[0])

        actual_str = tp.__str__()
        assert actual_str == "[3. 2. 3.]"

        N = 10
        samples = tp.sample_incremental(N, SampleMethod.random_uniform)
        assert len(samples) == N
        for transform in samples:
            assert transform.shape == (4, 4)
            assert_almost_equal(transform[1:3, 3], [2, 3])
            assert transform[0, 3] <= 4.0
            assert transform[0, 3] >= 2.0

    def test_x_and_z_tolerance(self):
        x = TolerancedNumber(2, 4, num_samples=3)
        z = TolerancedNumber(0, 3, nominal=1, num_samples=2)
        pos = [x, 2, z]
        rpy = [0, 0, 0]
        pp = TolEulerPt(pos, rpy)

        samples = pp.sample_grid()
        assert len(samples) == (3 * 2)

        # x samples [2, 3, 4], y samples [0, 3] and z fixed
        actual_pos = np.array([T[:3, 3] for T in samples])
        desired_pos = np.array(
            [[2, 2, 0], [2, 2, 3], [3, 2, 0], [3, 2, 3], [4, 2, 0], [4, 2, 3]]
        )
        assert_almost_equal(actual_pos, desired_pos)

        # rpy is set to zeros, so all rotation matrices should be unit matrices
        for T in samples:
            assert_almost_equal(T[:3, :3], np.eye(3))


class TestFreeOrientationPt:
    def test_create(self):
        point = FreeOrientationPt([1, 2, 3])

    def test_sample_incremental(self):
        point = FreeOrientationPt([1, 2, 3])
        samples = point.sample_incremental(5, SampleMethod.random_uniform)
        assert len(samples) == 5
        for T in samples:
            assert_almost_equal(T[:3, 3], np.array([1, 2, 3]))


class TestPathPointNumber:
    def test_create(self):
        a = PathPointNumber(5)
        assert a.nominal == 5
        assert a.discretize() == 5


class TestTolerancedNumber:
    def test_nominal_outside_bounds_error(self):
        with pytest.raises(ValueError) as info:
            TolerancedNumber(0, 1, nominal=1.5)
        # check whether the error message is present
        msg = "Nominal value must respect the bounds."
        assert msg in str(info.value)

    def test_get_initial_sampled_range(self):
        a = TolerancedNumber(0, 4, num_samples=5)
        a1 = a.discretize()
        d1 = [0, 1, 2, 3, 4]
        assert_almost_equal(a1, d1)
