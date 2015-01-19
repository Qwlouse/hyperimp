#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np

uniform = lambda x: x, lambda y: y


def logarithmic(base=None):
    if base is None:
        return lambda x: np.log(x), lambda y: np.exp(y)
    elif base == 10:
        return lambda x: np.log10(x), lambda y: np.power(10, y)
    elif base == 2:
        return lambda x: np.log2(x), lambda y: np.power(2, y)
    else:
        return lambda x: np.log(x) / np.log(base), lambda y: np.power(base, y)

