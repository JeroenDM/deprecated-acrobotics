#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.path import *

import pytest
import numpy as np
from numpy.testing import assert_almost_equal
from pyquaternion import Quaternion


class TestTolerancedNumber:
    def test_nominal_outside_bounds_error(self):
        with pytest.raises(ValueError) as info:
            a = TolerancedNumber(0, 1, nominal=1.5)
        # check whether the error message is present
        msg = "ValueError: Nominal value must respect the bounds"
        print(info)
        assert msg in str(info)

    def test_get_initial_sampled_range(self):
        a = TolerancedNumber(0, 4, samples=5)
        a1 = a.discretise()
        d1 = [0, 1, 2, 3, 4]
        assert_almost_equal(a1, d1)


class TestSamplers:
    def test_SO3_sampler(self):
        a = sample_SO3(rep="quat")
        assert len(a) == 10
        assert type(a[0]) is Quaternion

    def test_SO3_sampler_n(self):
        a = sample_SO3(n=15, rep="quat")
        assert len(a) == 15
        assert type(a[0]) is Quaternion

    def test_SO3_sampler_transform(self):
        a = sample_SO3(rep="transform")
        assert len(a) == 10
        assert a[0].shape == (4, 4)

    def test_SO3_sampler_rpy(self):
        a = sample_SO3(rep="rpy")
        assert len(a) == 10
        assert a[0].shape == (3,)


class TestFreeOrientationPt:
    def test_sampling(self):
        pt = FreeOrientationPt([1, 2, 3])
        a = pt.get_samples(5)
        assert a.shape == (5, 6)
