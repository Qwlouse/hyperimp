#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from hyperimp.utils import get_partitions, range_size, is_in_space


class MarginalizingFunction(object):
    def __init__(self, conf_space, tree):
        self.space = conf_space
        self.partitions = get_partitions(tree, self.space)

    def get_marginal(self, feat_indices=(), feat_values=()):
        marginal = 0
        size_of_theta_u = range_size(self.space, without=feat_indices)
        assert len(feat_indices) == len(feat_values)

        for p, val in self.partitions:
            if not all([p[i][0] <= v < p[i][1]
                        for i, v in zip(feat_indices, feat_values)]):
                continue
            marginal += val * range_size(p, without=feat_indices) /\
                        size_of_theta_u

        return marginal

    def get_marginal_var(self, feat_indices=(), feat_values=()):
        marginal = self.get_marginal(feat_indices, feat_values)
        var = 0
        size_of_theta_u = range_size(self.space, without=feat_indices)
        assert len(feat_indices) == len(feat_values)

        for p, val in self.partitions:
            if not all([p[i][0] <= v < p[i][1]
                        for i, v in zip(feat_indices, feat_values)]):
                continue
            var += (val - marginal) ** 2 * \
                   range_size(p, without=feat_indices) / size_of_theta_u

        return var


class MarginalizeOverDimsFunction(object):
    def __init__(self, conf_space, partitions, dim_idxs):
        self.space = conf_space
        self.partitions = partitions
        self.dim_idxs = dim_idxs

        size_of_theta_u = range_size(self.space, without=dim_idxs)
        self.partition_sizes = []
        self.variances = []

        for p, val in self.partitions:
            self.partition_sizes.append(range_size(p, without=dim_idxs) /
                                        size_of_theta_u)

    def get_marginal(self, dim_values):
        if isinstance(dim_values, np.ndarray) and dim_values.ndim == 2:
            marginal = np.zeros(dim_values.shape[1])
            assert dim_values.shape[0] == len(self.dim_idxs)
        else:
            marginal = 0
            assert len(dim_values) == len(self.dim_idxs)

        for i, (p, val) in enumerate(self.partitions):
            marginal += is_in_space(self.dim_idxs, dim_values, p) * val * \
                        self.partition_sizes[i]

        return marginal

    def get_marginal_var(self, dim_values):
        marginal = self.get_marginal_var(dim_values)
        var = np.zeros_like(marginal)

        for i, (p, val) in enumerate(self.partitions):
            var += is_in_space(self.dim_idxs, dim_values, p) * \
                   (val - marginal) ** 2 * self.partition_sizes[i]

        return var
