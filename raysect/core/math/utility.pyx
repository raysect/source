# cython: language_level=3

# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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

cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_index(ndarray x, double v):
    """
    Locates the lower array index of the array indicies that enclose the supplied value.

    bisection search

    expects a monotonically increasing ndarray as x, must be doubles, cannot be empty

    each array bin is defined [x[i], x[i+1]) is the bin corresponding to index i
    """

    cdef:
        double[::1] x_view
        int bottom_index
        int top_index
        int bisection_index

    x_view = x

    # check array ends before doing a costly bisection search
    if v < x_view[0]:

        # value is lower than the lowest value in the array
        return -1

    top_index = len(x) - 1
    if v >= x_view[top_index]:

        # value is above or equal to the highest value in the array
        return top_index

    # bisection search inside array range
    bottom_index = 0
    bisection_index = top_index / 2
    while (top_index - bottom_index) != 1:

        if v >= x_view[bisection_index]:

            bottom_index = bisection_index

        else:

            top_index = bisection_index

        bisection_index = (top_index + bottom_index) / 2

    return bottom_index


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrate(ndarray x, ndarray y, double x0, double x1):

    cdef:
        double[::1] x_view, y_view
        double integral_sum
        int top_index
        int lower_index
        int upper_index
        int index
        double y0, y1, m

    # invalid range
    if x1 <= x0:

        return 0.0

    # fast memoryview access to numpy arrays
    x_view = x
    y_view = y

    # identify array indicies that lie between requested values
    lower_index = find_index(x, x0) + 1
    upper_index = find_index(x, x1)

    # range is outside array limits
    top_index = len(x) - 1
    if lower_index > top_index or upper_index < 0:

        return 0.0

    if lower_index > upper_index:

        # both values lie inside the same array segment
        # the names lower_index and upper_index are now misnomers, they are swapped!
        m = (y_view[lower_index] - y_view[upper_index]) / (x_view[lower_index] - x_view[upper_index])
        y0 = m * (x0 - x_view[upper_index]) + y_view[upper_index]
        y1 = m * (x1 - x_view[upper_index]) + y_view[upper_index]

        # trapezium rule integration
        return 0.5 * (y0 + y1) * (x1 - x0)

    else:

        # add lower range partial cell contribution, if required
        if lower_index > 0:

            # linearly interpolate array cell at specified lower bound
            y0 = lerp(x_view[lower_index - 1], x_view[lower_index],
                      y_view[lower_index - 1], y_view[lower_index],
                      x0)

            # trapezium rule integration
            integral_sum = 0.5 * (y0 + y_view[lower_index]) * (x_view[lower_index] - x0)

        else:

            integral_sum = 0.0

        # sum up whole cell contributions
        for index in range(lower_index, upper_index):

            # trapezium rule integration
            integral_sum += 0.5 * (y_view[index] + y_view[index + 1]) * (x_view[index + 1] - x_view[index])

        # add upper range partial cell contribution, if required
        if upper_index < top_index:

            # linearly interpolate array cell at specified lower bound
            y1 = lerp(x_view[upper_index], x_view[upper_index + 1],
                      y_view[upper_index], y_view[upper_index + 1],
                      x1)

            # trapezium rule integration
            integral_sum += 0.5 * (y_view[upper_index] + y1) * (x1 - x_view[upper_index])

        return integral_sum
