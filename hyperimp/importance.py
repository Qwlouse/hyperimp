#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from hyperimp.marg_func import MarginalizingFunction
from hyperimp.utils import divide_global_space


def quantify_importance(configuration_space, tree, K=1):
    func = MarginalizingFunction(configuration_space, tree)
    global_divisions = divide_global_space(configuration_space, tree)
    f_avg = func.get_marginal()
    f_var = func.get_marginal_var()
    F = []
    #for k in range(K):
    for u in range(len(configuration_space)):
        var_u = 0
        for u_low, u_high in zip(global_divisions[u][:-1], global_divisions[u][1:]):
            a_u_theta_u = func.get_marginal([u], [u_low])
            f_u_theta_u = (a_u_theta_u - f_avg) # this is only true for K=1  (it needs dynamic programming later on)
            if (configuration_space[u][1] - configuration_space[u][0]) == 0:
                continue
            var_u += f_u_theta_u ** 2 * (u_high - u_low) / (configuration_space[u][1] - configuration_space[u][0])
        F.append(var_u / f_var)
    return F