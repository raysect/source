# cython: language_level=3

# Copyright (c) 2014-2017, Dr Alex Meakins, Raysect Project
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

"""
Integration utilities for regular grid emitters.
"""

from scipy.sparse import csc_matrix
from raysect.core.math.cython cimport find_index
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef object integrate_contineous(double[::1] x, object y, double x0, double x1, bint extrapolate=True):
    """
    Integrate the csc_matrix over its column axis in the specified range.

    :param double[::1] x: A memory view to a double array containing monotonically increasing
        values.
    :param csc_matrix y: A csc_matrix to integrate with the columns corresponding to
        the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.
    :param bool extrapolate: If True, the values of y outside the provided range
        will be equal to the values at the borders of this range (nearest-neighbour
        extrapolation), otherwise it will be zero. Defaults to `extrapolate=True`.

        :return: Integrated csc_matrix (one-column csc_matrix).
    """

    cdef:
        object integral_sum
        double weight
        int index, lower_index, upper_index, top_index, nvoxel

    nvoxel = y.shape[0]

    # invalid range
    if x1 <= x0:
        return csc_matrix((nvoxel, 1))

    # identify array indices that lie between requested values
    lower_index = find_index(x, x0) + 1
    upper_index = find_index(x, x1)

    # are both points below the bottom of the array?
    if upper_index == -1:

        if extrapolate:
            # extrapolate from first array value (nearest-neighbour)
            return y[:, 0] * (x1 - x0)
        # return zero matrix if extrapolate is set to False
        return csc_matrix((nvoxel, 1))

    # are both points beyond the top of the array?
    top_index = x.shape[0] - 1
    if lower_index > top_index:

        if extrapolate:
            # extrapolate from last array value (nearest-neighbour)
            return y[:, top_index] * (x1 - x0)
        # return zero matrix if extrapolate is set to False
        return csc_matrix((nvoxel, 1))

    # numerically integrate array
    if lower_index > upper_index:

        # both values lie inside the same array segment
        # the names lower_index and upper_index are now misnomers, they are swapped!
        weight = (0.5 * (x1 + x0) - x[upper_index]) / (x[lower_index] - x[upper_index])

        # trapezium rule integration
        return (y[:, upper_index] + weight * (y[:, lower_index] - y[:, upper_index])) * (x1 - x0)

    else:

        integral_sum = csc_matrix((nvoxel, 1))

        if lower_index == 0:

            # add contribution from point below array
            integral_sum += y[:, 0] * (x[0] - x0)

        else:

            # add lower range partial cell contribution
            weight = (x0 - x[lower_index - 1]) / (x[lower_index] - x[lower_index - 1])

            # trapezium rule integration
            integral_sum += (0.5 * (x[lower_index] - x0)) * (y[:, lower_index - 1] + y[:, lower_index] +
                                                             weight * (y[:, lower_index] - y[:, lower_index - 1]))

        # sum up whole cell contributions
        for index in range(lower_index, upper_index):

            # trapezium rule integration
            integral_sum += 0.5 * (y[:, index] + y[:, index + 1]) * (x[index + 1] - x[index])

        if upper_index == top_index:

            # add contribution from point above array
            integral_sum += y[:, top_index] * (x1 - x[top_index])

        else:

            # add upper range partial cell contribution
            weight = (x1 - x[upper_index]) / (x[upper_index + 1] - x[upper_index])

            # trapezium rule integration
            integral_sum += (0.5 * (x1 - x[upper_index])) * (2 * y[:, upper_index] + weight * (y[:, upper_index + 1] - y[:, upper_index]))

        return integral_sum


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef object integrate_delta_function(double[::1] x, object y, double x0, double x1):
    """
    Integrate delta-function-like csc_matrix over its column axis in the specified range.

    :param double[::1] x: A memory view to a double array containing monotonically increasing
        values.
    :param csc_matrix y: A delta-function-like csc_matrix to integrate with the columns
        corresponding to the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.

    :return: Integrated csc_matrix (one-column csc_matrix).
    """
    cdef:
        object integral_sum
        int i, nvoxel, nspec

    nvoxel = y.shape[0]
    nspec = y.shape[1]

    # invalid range
    if x1 <= x0:
        return csc_matrix((nvoxel, 1))

    integral_sum = csc_matrix((nvoxel, 1))

    for i in range(nspec):
        if x0 <= x[i] < x1:
            integral_sum += y[:, i]

    return integral_sum
