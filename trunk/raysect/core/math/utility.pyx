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
cdef inline int find_index(ndarray x, double v):
    """
    Locates the lower index or the range that contains the specified value.

    This function performs a fast bisection search to identify the index range
    (bin) that encloses the specified value. The lower index of the range is
    returned. This function expects a monotonically increasing ndarray for x.
    The array type must be double and may not be empty.

    Each array bin has the defined range [x[i], x[i+1]) where i is the index of
    the bin.

    If the value lies below the range of the array this function will return an
    index of -1. If the value lies above the range of the array then the last
    index of the array will be returned.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param ndarray x: An array containing monotonically increasing values.
    :param double v: The value to search for.
    :return: The lower index f the bin containing the search value.
    :rtype: int
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


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double interpolate(ndarray x, ndarray y, double p):
    """
    Linearly interpolates sampled data onto the specified point.

    This function performs a linear interpolation of sampled function data
    on to x = p. Outside the bounds of the array the value is taken to be the
    end value of the array closest to the requested point (nearest-neighbour
    extrapolation).

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param ndarray x: An array containing monotonically increasing values.
    :param ndarray y: An array of sample values corresponding to the x array points.
    :param double p: The x point for which an interpolated y value is required.
    :return: The linearly interpolated y value at point p.
    :rtype: double
    """

    cdef:
        int index, top_index
        double[::1] x_view, y_view

    # obtain memory views
    x_view = x
    y_view = y

    index = find_index(x, p)

    # point is below array limits
    if index == -1:

        return y_view[0]

    # wavelength is above array limits
    top_index = x_view.shape[0] - 1
    if index == top_index:

        return y_view[top_index]

    # interpolate inside array
    return lerp(x_view[index], x_view[index + 1], y_view[index], y_view[index + 1], p)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double integrate(ndarray x, ndarray y, double x0, double x1):
    """
    Integrates a linearly interpolated function between two points.

    This function performs a trapezium rule integration of the sampled function
    between point x0 and point x1. Outside the bounds of the array the function
    value is taken to be the end value of the array closest to the requested
    point (nearest-neighbour extrapolation).

    If x1 < x0 the integral range is treated as null and zero is returned.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param ndarray x: An array containing monotonically increasing values.
    :param ndarray y: An array of sample values corresponding to the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.
    :return: Integral between x0 and x1
    :rtype: double
    """

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

    # identify array indices that lie between requested values
    lower_index = find_index(x, x0) + 1
    upper_index = find_index(x, x1)

    # are both points below the bottom of the array?
    if upper_index == -1:

        # extrapolate from first array value (nearest-neighbour)
        return y[0] * (x1 - x0)

    # are both points beyond the top of the array?
    top_index = len(x) - 1
    if lower_index > top_index:

        # extrapolate from last array value (nearest-neighbour)
        return y[top_index] * (x1 - x0)

    # fast memoryview access to numpy arrays
    x_view = x
    y_view = y

    # numerically integrate array
    if lower_index > upper_index:

        # both values lie inside the same array segment
        # the names lower_index and upper_index are now misnomers, they are swapped!
        m = (y_view[lower_index] - y_view[upper_index]) / (x_view[lower_index] - x_view[upper_index])
        y0 = m * (x0 - x_view[upper_index]) + y_view[upper_index]
        y1 = m * (x1 - x_view[upper_index]) + y_view[upper_index]

        # trapezium rule integration
        return 0.5 * (y0 + y1) * (x1 - x0)

    else:

        integral_sum = 0.0

        if lower_index == 0:

            # add contribution from point below array
            integral_sum += y_view[0] * (x_view[0] - x0)

        else:

            # add lower range partial cell contribution
            y0 = lerp(x_view[lower_index - 1], x_view[lower_index],
                      y_view[lower_index - 1], y_view[lower_index],
                      x0)

            # trapezium rule integration
            integral_sum += 0.5 * (y0 + y_view[lower_index]) * (x_view[lower_index] - x0)

        # sum up whole cell contributions
        for index in range(lower_index, upper_index):

            # trapezium rule integration
            integral_sum += 0.5 * (y_view[index] + y_view[index + 1]) * (x_view[index + 1] - x_view[index])

        if upper_index == top_index:

            # add contribution from point above array
            integral_sum += y_view[top_index] * (x1 - x_view[top_index])

        else:

            # add upper range partial cell contribution
            y1 = lerp(x_view[upper_index], x_view[upper_index + 1],
                      y_view[upper_index], y_view[upper_index + 1],
                      x1)

            # trapezium rule integration
            integral_sum += 0.5 * (y_view[upper_index] + y1) * (x1 - x_view[upper_index])

        return integral_sum


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double average(ndarray x, ndarray y, double x0, double x1):
    """
    Returns the average value of a linearly interpolated function between two
    points.

    Outside the bounds of the array the function value is taken to be the end
    value of the array closest to the requested point (nearest-neighbour
    extrapolation).

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param ndarray x: An array containing monotonically increasing values.
    :param ndarray y: An array of sample values corresponding to the x array points.
    :param double x0: First point.
    :param double x1: Second point.
    :return: Mean value between x0 and x1
    :rtype: double
    """

    cdef:
        double[::1] x_view, y_view
        int index, top_index
        double temp

    if x0 == x1:

        # single point, just sample function

        # fast memoryview access to numpy arrays
        x_view = x
        y_view = y

        index = find_index(x, x0)

        # is point below array?
        if index == -1:

            return y_view[0]

        top_index = len(x) - 1

        # is point above array?
        if index == top_index:

            return y_view[top_index]

        # point is within array
        return lerp(x_view[index], x_view[index + 1],
                    y_view[index], y_view[index + 1],
                    x0)

    else:

        # ensure x0 is always lower than x1
        if x1 < x0:

            temp = x0
            x0 = x1
            x1 = temp

        return integrate(x, y, x0, x1) / (x1 - x0)
