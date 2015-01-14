#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from hyperimp.utils import range_size, divide_space


def almost_equal(a, b):
    return abs(a - b) < 1e-10


@pytest.fixture
def space():
    return np.array([
        [1, 2],        # 1
        [-2, 5],       # 7
        [0, 0.1]       # 0.1
    ])


def test_range_size(space):
    assert almost_equal(range_size(space), 0.7)


def test_range_size_without(space):
    assert almost_equal(range_size(space, without=[0]), 0.7)
    assert almost_equal(range_size(space, without=[1]), 0.1)
    assert almost_equal(range_size(space, without=[2]), 7)
    assert almost_equal(range_size(space, without=[0, 1]), 0.1)
    assert almost_equal(range_size(space, without=[1, 2]), 1.)


def test_divide_space(space):
    lspace, rspace = divide_space(space, 1, 0)
    assert np.all(lspace[0] == rspace[0])
    assert np.all(lspace[1] == [-2, 0])
    assert np.all(rspace[1] == [0, 5])
    assert np.all(lspace[2] == rspace[2])