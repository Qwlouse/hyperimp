#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from hyperimp.utils import get_configuration_space, get_partitions
from hyperimp.importance import quantify_importance
from hyperimp.marg_func import MarginalizeOverDimsFunction


def get_forest(parameters, performance, n_trees=100):
    clf = RandomForestRegressor(n_estimators=n_trees, max_depth=None,
                                min_samples_split=1)
    clf = clf.fit(parameters, performance)
    return clf


def analyse(params, perf, K=1, param_names=None, n_trees=100):
    # train predictor
    clf = get_forest(params, perf, n_trees)

    configuration_space = get_configuration_space(params)
    var_fracs = []
    partitionings = []
    key_combinations = []
    for est in clf.estimators_:
        tree = est.tree_
        fu = quantify_importance(configuration_space, tree, K=K)
        key_combinations = sorted(fu.keys())
        var_fracs.append([fu[k] for k in key_combinations])
        partitionings.append(get_partitions(tree, configuration_space))

    result = {
        'marginals': {},
        'avg_stds': {}
    }

    margs = np.array(var_fracs)
    mean_vars = margs.mean(0)
    if param_names is None:
        param_names = ['X%d' % i for i in range(len(mean_vars))]

    main_vars = sorted(zip(key_combinations, mean_vars), key=lambda x: -x[1])
    for i, (keys, mv) in enumerate(main_vars):
        name = " × ".join([param_names[j] for j in keys])
        print("%d)" % i, name, "= %.2f%%" % (mv * 100))
        result['marginals'][name] = mv

    for u, name in enumerate(param_names):
        low, high = configuration_space[u]
        X = np.arange(low, high, (high - low) / 100)
        margs = []
        for part in partitionings:
            f_u = MarginalizeOverDimsFunction(configuration_space, part, [u])
            margs.append(f_u.get_marginal([X]).flatten())
        result['avg_stds'][name] = X, np.mean(margs, 0), np.std(margs, 0)

    return result

