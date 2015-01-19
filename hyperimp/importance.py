#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from itertools import combinations, product

from hyperimp.marg_func import MarginalizingFunction, \
    MarginalizeOverDimsFunction
from hyperimp.utils import divide_global_space


def get_fu(conf_space, partitions, low_coords, us):
    func_u = MarginalizeOverDimsFunction(conf_space, partitions, us)
    f_u = func_u.get_marginal(low_coords)

    for l in range(len(us)):
        for idxs, w in zip(combinations(range(len(us)), l),
                           combinations(us, l)):
            f_u -= get_fu(conf_space, partitions, low_coords[list(idxs), :], w)

    return f_u


def quantify_importance(configuration_space, tree, K=1):
    func = MarginalizingFunction(configuration_space, tree)
    global_divisions = divide_global_space(configuration_space, tree)
    f_var = func.get_marginal_var()
    F = {}
    for k in range(1, K + 1):
        for us in combinations(range(len(configuration_space)), k):
            u_range = np.product([(configuration_space[u][1] -
                                   configuration_space[u][0]) for u in us])
            if u_range == 0:
                continue

            # get the "lower left corners" of all the regions
            low_coords = np.array(list(product(*[global_divisions[u][:-1]
                                                 for u in us]))).T
            # get the "upper right corners" of all the regions
            high_coords = np.array(list(product(*[global_divisions[u][1:]
                                                  for u in us]))).T

            # calculate the sizes of all the regions
            sizes = np.product(high_coords - low_coords, axis=0)

            f_u = get_fu(configuration_space, func.partitions, low_coords, us)

            var_u = np.sum(f_u ** 2 * sizes / u_range)
            F[us] = var_u / f_var
    return F
