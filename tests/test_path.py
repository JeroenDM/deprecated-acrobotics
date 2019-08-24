#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.path.toleranced_number import (
    TolerancedNumber,
    FixedNumber,
    TolerancedQuaternion,
)
from acrobotics.path.path_pt import TolEulerPt, PathPt, TolPositionPt, TolQuatPt
from acrobotics.samplers import SampleMethod
from acrobotics.robot import Robot, IKResult
from acrobotics.types import SamplingType
from acrobotics.planning_setting import PlanningSetting

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from acrobotics.pyquat_extended import QuaternionExtended as Quaternion

IK_RESULT = IKResult(True, [np.ones(6), 2 * np.ones(6)])


class DummyRobot(Robot):
    def __init__(self, is_colliding=False):
        self.is_colliding = is_colliding
        self.ndof = 6

    def is_in_collision(self, joint_position, scene=None):
        return self.is_colliding

    def ik(self, transform):
        return IK_RESULT


class TestPathPt:
    def test_calc_ik(self):
        samples = [[], []]
        joint_solutions = PathPt._calc_ik(DummyRobot(), samples)
        assert len(joint_solutions) == 4

        assert_almost_equal(joint_solutions[0], IK_RESULT.solutions[0])
        assert_almost_equal(joint_solutions[1], IK_RESULT.solutions[1])
        assert_almost_equal(joint_solutions[2], IK_RESULT.solutions[0])
        assert_almost_equal(joint_solutions[3], IK_RESULT.solutions[1])


class TestTolPositionPt:
    def test_create(self):
        TolPositionPt([1, 2, 3], Quaternion())

    def test_sample_grid(self):
        x = TolerancedNumber(0, 1, num_samples=2)
        y = TolerancedNumber(0, 1, num_samples=2)
        z = 1
        q = Quaternion()

        point = TolPositionPt([x, y, z], q)
        grid = point.sample_grid()
        position_samples = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        for pos, T in zip(position_samples, grid):
            assert_almost_equal(pos, T[:3, 3])


class TestTolQuatPt:
    def test_create(self):
        q = TolerancedQuaternion(Quaternion(), 0.5)
        TolQuatPt([1, 2, 3], q)

    def test_sample_incremental(self):
        method = SampleMethod.random_uniform
        distance = 0.1
        tolquat = TolerancedQuaternion(Quaternion(), distance)
        point = TolQuatPt([1, 2, 3], tolquat)
        samples = point.sample_incremental(100, method)

        for tf in samples:
            newquat = Quaternion(matrix=tf)
            assert Quaternion.distance(tolquat.quat, newquat) <= distance

    def test_to_transform(self):
        distance = 0.1
        tolquat = TolerancedQuaternion(Quaternion(), distance)
        point = TolQuatPt([1, 2, 3], tolquat)
        quat = Quaternion([1, 0, 0, 0])

        tf = point.to_transform(quat)
        assert_almost_equal(
            [[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], tf
        )


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

    def setting_generator(self, sampling_type):
        if sampling_type == SamplingType.GRID:
            return PlanningSetting(
                SamplingType.GRID, SampleMethod.random_uniform, 10, 10, 10
            )
        if sampling_type == SamplingType.INCREMENTAL:
            return PlanningSetting(
                SamplingType.INCREMENTAL, SampleMethod.random_uniform, 10, 10, 10, 2
            )
        if sampling_type == SamplingType.MIN_INCREMENTAL:
            return PlanningSetting(
                SamplingType.MIN_INCREMENTAL, SampleMethod.random_uniform, 10, 10, 10, 2
            )

    def test_to_joint_solutions(self):
        for sampling_type in SamplingType:
            x = TolerancedNumber(2, 4, num_samples=3)
            pos = [x, 2, 3]
            rpy = [4, 5, 6]
            euler_pt = TolEulerPt(pos, rpy)
            robot = DummyRobot(is_colliding=True)
            settings = self.setting_generator(sampling_type)
            if sampling_type is not SamplingType.MIN_INCREMENTAL:
                joint_solutions = euler_pt.to_joint_solutions(robot, settings)
                assert len(joint_solutions) == 0
            else:
                with pytest.raises(Exception) as info:
                    joint_solutions = euler_pt.to_joint_solutions(robot, settings)
                msg = "Maximum iterations reached in to_joint_solutions."
                assert msg in str(info.value)


class TestFixedNumber:
    def test_create(self):
        a = FixedNumber(5)
        assert a.nominal == 5
        assert a.discretize() == 5


class TestTolerancedNumber:
    def test_get_initial_sampled_range(self):
        a = TolerancedNumber(0, 4, num_samples=5)
        a1 = a.discretize()
        d1 = [0, 1, 2, 3, 4]
        assert_almost_equal(a1, d1)

    def test_calc_reduced_bounds(self):
        x = TolerancedNumber(2, 4)
        x.reduce_bounds(3, 2)
        assert_almost_equal(x.lower, 2.5)
        assert_almost_equal(x.upper, 3.5)

        x = TolerancedNumber(-4, -1)
        x.reduce_bounds(-3, 10)
        assert_almost_equal(x.lower, -3.15)
        assert_almost_equal(x.upper, -2.85)

        x = TolerancedNumber(-3, 7)
        x.reduce_bounds(4, 5)
        assert_almost_equal(x.lower, 3)
        assert_almost_equal(x.upper, 5)
