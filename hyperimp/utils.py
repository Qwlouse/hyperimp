#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np


def range_size(space, without=None):
    """
    Calculate the range-size (denoted || space || in the paper) for a
    given space.

    :param space: Two dimensional array storing the low and high limits of the
        space
    :type space: ndarray
    :param without: list of dimension indices to omit
    :type without: list[int]
    :return: range-size of space
    :rtype: float
    """

    total_size = np.prod(space[:, 1] - space[:, 0])
    if without is not None:
        total_size /= np.prod(space[without, 1] - space[without, 0])
    return total_size


def divide_space(space, dim_idx, threshold):
    """
    Divide a given space (list of intervals) in dimension with index
    `dim_idx` along `threshold` into two spaces: a left and a right space.

    :param space: configuration space as a list of intervals
    :type space: ndarray
    :param dim_idx: index of the dimension that should be split
    :type dim_idx: int
    :param threshold: the value at which the dimension should be split
    :type threshold: float
    :return: the left- and right-space after the split
    :rtype: (ndarray, ndarray)
    """
    low, high = space[dim_idx]
    assert low <= threshold <= high

    left_space = space.copy()
    left_space[dim_idx] = low, threshold

    right_space = space.copy()
    right_space[dim_idx] = threshold, high

    return left_space, right_space


def get_partitions(tree, space, node_idx=0):
    """
    Get the list of partitions (subspaces) that the given tree splits the given
    space into.
    :param tree: A regression tree
    :type tree: sklearn.tree._tree.Tree
    :param space: The input space
    :type space: ndarray
    :param node_idx: the node from which to start the splitting
        (default = root)
    :type node_idx: int
    :return: list of partitions
    :rtype: list[ndarray]
    """
    left_child = tree.children_left[node_idx]
    right_child = tree.children_right[node_idx]
    if left_child == right_child:  # == TREE_LEAF == -1
        value = float(tree.value[node_idx])
        return [(space, value)]

    left_space, right_space = divide_space(space, tree.feature[node_idx],
                                           tree.threshold[node_idx])

    l_part = get_partitions(tree, left_space, left_child)
    r_part = get_partitions(tree, right_space, right_child)

    return l_part + r_part