#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from hyperimp.utils import get_partitions, range_size


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
