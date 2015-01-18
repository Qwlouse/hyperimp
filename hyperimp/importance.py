#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from hyperimp.marg_func import MarginalizingFunction, \
    MarginalizeOverDimsFunction
from hyperimp.utils import divide_global_space


def quantify_importance(configuration_space, tree, K=1):
    func = MarginalizingFunction(configuration_space, tree)
    global_divisions = divide_global_space(configuration_space, tree)
    f_avg = func.get_marginal()
    f_var = func.get_marginal_var()
    F = []
    #for k in range(K):
    for u in range(len(configuration_space)):
        u_range = (configuration_space[u][1] - configuration_space[u][0])
        if u_range == 0:
            continue
        func_u = MarginalizeOverDimsFunction(configuration_space,
                                             func.partitions, [u])
        a_u = func_u.get_marginal([global_divisions[u][:-1]])
        f_u = a_u - f_avg  # this is only true for K=1
        #                    (it needs dynamic programming later on)
        lows = np.array(global_divisions[u][:-1])
        highs = np.array(global_divisions[u][1:])
        var_u = np.sum(f_u ** 2 * (highs - lows) / u_range)
        F.append(var_u / f_var)
    return F
