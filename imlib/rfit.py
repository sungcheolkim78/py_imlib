#!/usr/local/bin/python3
"""
rfit.py - functions for robust fit

date: 20180331 - derived from _im_utils
date: 20180723 - add more functions
date: 20190930 - optimize using numba
"""

from scipy.optimize import minimize
import numpy as np
import math
from .fmin import fmin
from numba import njit, vectorize, float64

__author__ = 'Sung-Cheol Kim'
__version__ = '1.0.0'


@njit(fastmath=True)
def inverseabs(k, x):
    return k[2] / (1.0 + np.abs((x - k[0]) / k[1]))


@njit(fastmath=True)
def gaussian(k, x):
    """ gaussian function
    k - coefficient array, x - values """
    return k[2] * np.exp( -(x - k[0]) * (x - k[0]) / (2 * k[1] * k[1])) + k[3]


@njit(fastmath=True)
def gaussian2(k, x):
    """ gaussian function
    k - coefficient array, x - values """
    return k[4] * np.exp(- (x - k[0])**2 / (2 * k[2]**2)) \
         + k[5] * np.exp(- (x - k[1])**2 / (2 * k[3]**2)) + k[6]


@njit(fastmath=True)
def gaussian3(k, x):
    """ gaussian function
    k - coefficient array, x - values """
    return k[0] * np.exp(-(x - k[1])**2 / (2 * k[2]**2)) \
         + k[3] * np.exp(-(x - k[4])**2 / (2 * k[5]**2)) \
         + k[6] * np.exp(-(x - k[7])**2 / (2 * k[8]**2)) + k[9]


@njit(fastmath=True)
def line(k, x):
    """ line function """
    return k[0] * x + k[1]


@njit(fastmath=True)
def poly2(k, x):
    """ line function """
    return k[0] * x * x + k[1] * x + k[2]


@njit(fastmath=True)
def erf(k, x):
    """ error function """
    return k[2] * math.erf((x - k[0])/(np.sqrt(2)*k[1])) + k[3]*x + k[4]


def gcdf(k, x):
    """ cumulative gaussian distribution """
    return [ 0.5*(1.0 + math.erf((t - k[0])/(np.sqrt(2)*k[1])))+k[2]*t+k[3] for t in x ]


def loss(k, x, y, f, nu):
    """ optimization function
    k - coefficients
    x, y - values
    f - function
    nu - normalization factor """
    res = y - f(k, x)
    return np.sum(np.log(1 + res**2 / nu))


def robust_gaussian_fit(x, y, nu=1.0, initial=[1.0, 0.0, 1.0, 0.0], debug=False):
    """ robust fit using log loss function - gaussian """
    return minimize(loss, initial, args=(x, y, gaussian, nu), method='Nelder-Mead', options={'disp':debug})


def robust_gaussian2_fit(x, y, nu=1.0, initial=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0], debug=False):
    """ robust fit using log loss function - double gaussian """
    return minimize(loss, initial, args=(x, y, gaussian2, nu), method='Nelder-Mead', options={'disp':debug})


def robust_gaussian3_fit(x, y, nu=1.0, initial=[1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0], debug=False):
    """ robust fit using log loss function - triple gaussian """
    return minimize(loss, initial, args=(x, y, gaussian3, nu), method='Nelder-Mead', options={'disp':debug})


def robust_line_fit(x, y, nu=1.0, initial=[0.1, 0.0], debug=False):
    """ robust fit using log loss function """
    return minimize(loss, initial, args=(x, y, line, nu), method='Nelder-Mead', options={'disp':debug})
    #return fmin(loss, initial, args=(x, y, line, nu), disp=debug)


def robust_poly2_fit(x, y, nu=1.0, initial=[0.1, 0.1, 0.0], debug=False):
    """ robust fit using log loss function """
    return minimize(loss, initial, args=(x, y, poly2, nu), method='Nelder-Mead', options={'disp':debug})


def robust_gcdf_fit(x, y, nu=1.0, initial=[0.0, 1.0, 0.001, 0.001], debug=False):
    """ robust fit using log loss function """
    return minimize(loss, initial, args=(x, y, gcdf, nu), method='Nelder-Mead', options={'disp':debug})


def robust_inverseabs_fit(x, y, nu=1.0, initial=[1.0, 0.0, 1.0], debug=False):
    """ robust fit using log loss function """
    return minimize(loss, initial, args=(x, y, inverseabs, nu), method='Nelder-Mead', options={'disp':debug})

# vim:foldlevel=0
