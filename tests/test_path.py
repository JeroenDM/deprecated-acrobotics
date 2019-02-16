#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from acrobotics.path import TolerancedNumber, TrajectoryPt, sample_SO3, FreeOrientationPt

import pytest
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal
from pyquaternion import Quaternion

class TestTolerancedNumber():
    def test_nominal_outside_bounds_error(self):
        with pytest.raises(ValueError) as info:
            a = TolerancedNumber(1.5, 0, 1)
        # check whether the error message is present
        msg = "nominal value must respect the bounds"
        assert(msg in str(info))
    
    def test_get_initial_sampled_range(self):
        a = TolerancedNumber(2, 0, 4, samples=5)
        a1 = a.range
        d1 = [ 0, 1, 2, 3, 4]
        assert_almost_equal(a1, d1)


class TestSamplers:
    def test_SO3_sampler(self):
        a = sample_SO3()
        assert(len(a) == 10)
        assert(type(a[0]) is Quaternion)
    
    def test_SO3_sampler_n(self):
        a = sample_SO3(n=15)
        assert(len(a) == 15)
        assert(type(a[0]) is Quaternion)
    
    def test_SO3_sampler_transform(self):
        a = sample_SO3(rep='transform')
        assert(len(a) == 10)
        assert(a[0].shape == (4, 4))
    
    def test_SO3_sampler_rpy(self):
        a = sample_SO3(rep='rpy')
        assert(len(a) == 10)
        assert(a[0].shape == (3, ))

class TestFreeOrientationPt:
    def test_sampling(self):
        pt = FreeOrientationPt([1, 2, 3])
        a = pt.get_samples(5)
        assert(a.shape == (5, 6))