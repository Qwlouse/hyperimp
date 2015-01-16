#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import pytest
from hyperimp.utils import range_size, divide_space, is_in_space


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


def test_is_in_space(space):
    assert is_in_space([0], [1.5], space)
    assert not is_in_space([0], [.5], space)

    assert is_in_space([1], [0.], space)
    assert not is_in_space([1], [-3], space)

    assert is_in_space([2], [.05], space)
    assert not is_in_space([2], [.5], space)


def test_is_in_space_multiple(space):
    assert np.all(is_in_space([0], [-1, 0, 1, 2, 3], space) ==
                  [False, False, True, False, False])

    assert np.all(is_in_space([1], [-3, -2, -1, 0, 1, 2, 3], space) ==
                  [False, True, True, True, True, True, True])

    assert np.all(is_in_space([2], [-.05, 0., 0.05, .1], space) ==
                  [False, True, True, False])


def test_is_in_space_multimulti(space):
    assert np.all(is_in_space([0, 1], np.array([[-1, 0, 1, 1.5], [-3, 3, 6, 0.5]]), space) ==
                  [False, False, False, True])