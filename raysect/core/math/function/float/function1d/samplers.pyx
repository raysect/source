# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from numpy import asarray, ascontiguousarray, empty, linspace
from .base cimport Function1D
from .autowrap cimport autowrap_function1d
cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple sample1d(object function, double x_min, double x_max, int x_samples):
    """
    Samples Function1D over given range.
    
    :param function: the function to sample. a Function1D object, or a Python function 
    :param x_min: minimum value for sample range
    :param x_max: maximum value for sample range
    :param x_samples: number of samples between x_min and x_max, where endpoints are included
    :return: a tuple of sampled x points and respective function samples (x, f)
    """
    cdef:
        double[::1] x_v, f_v
        int i, samples
        Function1D func

    if x_min > x_max:
        raise ValueError(f"x_min ({x_min}) argument cannot be greater than x_max ({x_max})")

    if x_samples < 1:
        raise ValueError("The argument x_samples must be >= 1")


    # ensures that func is of type Function1D. I.e. if 'function' argument was Python function, it'll get autowrapped
    # into Function1D object
    func = autowrap_function1d(function)

    x = linspace(x_min, x_max, x_samples)
    f = empty(x_samples)

    # use memory views
    x_v = x
    f_v = f

    for i in range(x_samples):
        f_v[i] = func.evaluate(x_v[i])

    return x, f


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray sample1d_points(object function, object x_points):
    """
    Samples Function1D in given x points.
    
    :param function: the function to sample. a Function1D object, or a Python function
    :param x_points: array containing the points at which the function is sampled.
    :return: an array containing the sampled values of the given function.
    """
    cdef:
        double[::1] x_view, v_view
        int i, num_samples
        Function1D fun

    x_points = ascontiguousarray(x_points, dtype=float)

    # ensures that func is of type Function1D. I.e. if 'function' argument was Python function, it'll get autowrapped
    # into Function1D object
    fun = autowrap_function1d(function)
    num_samples = len(x_points)

    f = empty(num_samples)

    # memory views
    x_v = x_points
    f_v = f

    for i in range(num_samples):
        f_v[i] = fun.evaluate(x_v[i])

    return f