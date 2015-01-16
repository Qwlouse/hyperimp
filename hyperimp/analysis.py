#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
from sklearn.ensemble import RandomForestRegressor


def get_forest(parameters, performance, n_trees=100):
    clf = RandomForestRegressor(n_estimators=n_trees, max_depth=None,
                                min_samples_split=1)
    clf = clf.fit(parameters, performance)
    return clf

