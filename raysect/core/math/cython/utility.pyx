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

from libc.math cimport sqrt
cimport cython

#TODO: Write unit tests!

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int find_index(double[::1] x, double v) nogil:
    """
    Locates the lower index or the range that contains the specified value.

    This function performs a fast bisection search to identify the index range
    (bin) that encloses the specified value. The lower index of the range is
    returned. This function expects a monotonically increasing array for x. The
    array must be a double array and may not be empty.

    Each array bin has the defined range [x[i], x[i+1]) where i is the index of
    the bin.

    If the value lies below the range of the array this function will return an
    index of -1. If the value lies above the range of the array then the last
    index of the array will be returned.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param double[::1] x: A memory view to a double array containing monotonically increasing values.
    :param double v: The value to search for.
    :return: The lower index f the bin containing the search value.
    :rtype: int
    """

    cdef:
        int bottom_index
        int top_index
        int bisection_index

    # check array ends before doing a costly bisection search
    if v < x[0]:

        # value is lower than the lowest value in the array
        return -1

    top_index = x.shape[0] - 1
    if v >= x[top_index]:

        # value is above or equal to the highest value in the array
        return top_index

    # bisection search inside array range
    bottom_index = 0
    bisection_index = top_index / 2
    while (top_index - bottom_index) != 1:
        if v >= x[bisection_index]:
            bottom_index = bisection_index
        else:
            top_index = bisection_index
        bisection_index = (top_index + bottom_index) / 2
    return bottom_index


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double interpolate(double[::1] x, double[::1] y, double p) nogil:
    """
    Linearly interpolates sampled data onto the specified point.

    This function performs a linear interpolation of sampled function data
    on to x = p. Outside the bounds of the array the value is taken to be the
    end value of the array closest to the requested point (nearest-neighbour
    extrapolation).

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param double[::1] x: A memory view to a double array containing monotonically increasing values.
    :param double[::1] y: A memory view to a double array of sample values corresponding to the x array points.
    :param double p: The x point for which an interpolated y value is required.
    :return: The linearly interpolated y value at point p.
    :rtype: double
    """

    cdef:
        int index, top_index

    index = find_index(x, p)

    # point is below array limits
    if index == -1:
        return y[0]

    # wavelength is above array limits
    top_index = x.shape[0] - 1
    if index == top_index:
        return y[top_index]

    # interpolate inside array
    return lerp(x[index], x[index + 1], y[index], y[index + 1], p)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double integrate(double[::1] x, double[::1] y, double x0, double x1) nogil:
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

    :param double[::1] x: A memory view to a double array containing monotonically increasing values.
    :param double[::1] y: A memory view to a double array of sample values corresponding to the x array points.
    :param double x0: Start point of integration.
    :param double x1: End point of integration.
    :return: Integral between x0 and x1.
    :rtype: double
    """

    cdef:
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
    top_index = x.shape[0] - 1
    if lower_index > top_index:

        # extrapolate from last array value (nearest-neighbour)
        return y[top_index] * (x1 - x0)

    # numerically integrate array
    if lower_index > upper_index:

        # both values lie inside the same array segment
        # the names lower_index and upper_index are now misnomers, they are swapped!
        m = (y[lower_index] - y[upper_index]) / (x[lower_index] - x[upper_index])
        y0 = m * (x0 - x[upper_index]) + y[upper_index]
        y1 = m * (x1 - x[upper_index]) + y[upper_index]

        # trapezium rule integration
        return 0.5 * (y0 + y1) * (x1 - x0)

    else:

        integral_sum = 0.0

        if lower_index == 0:

            # add contribution from point below array
            integral_sum += y[0] * (x[0] - x0)

        else:

            # add lower range partial cell contribution
            y0 = lerp(x[lower_index - 1], x[lower_index],
                      y[lower_index - 1], y[lower_index],
                      x0)

            # trapezium rule integration
            integral_sum += 0.5 * (y0 + y[lower_index]) * (x[lower_index] - x0)

        # sum up whole cell contributions
        for index in range(lower_index, upper_index):

            # trapezium rule integration
            integral_sum += 0.5 * (y[index] + y[index + 1]) * (x[index + 1] - x[index])

        if upper_index == top_index:

            # add contribution from point above array
            integral_sum += y[top_index] * (x1 - x[top_index])

        else:

            # add upper range partial cell contribution
            y1 = lerp(x[upper_index], x[upper_index + 1],
                      y[upper_index], y[upper_index + 1],
                      x1)

            # trapezium rule integration
            integral_sum += 0.5 * (y[upper_index] + y1) * (x1 - x[upper_index])

        return integral_sum


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double average(double[::1] x, double[::1] y, double x0, double x1) nogil:
    """
    Returns the average value of a linearly interpolated function between two
    points.

    Outside the bounds of the array the function value is taken to be the end
    value of the array closest to the requested point (nearest-neighbour
    extrapolation).

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param double[::1] x: A memory view to a double array containing monotonically increasing values.
    :param double[::1] y: A memory view to a double array of sample values corresponding to the x array points.
    :param double x0: First point.
    :param double x1: Second point.
    :return: Mean value between x0 and x1.
    :rtype: double
    """

    cdef:
        int index, top_index
        double temp

    if x0 == x1:

        # single point, just sample function
        index = find_index(x, x0)

        # is point below array?
        if index == -1:
            return y[0]

        top_index = x.shape[0] - 1

        # is point above array?
        if index == top_index:
            return y[top_index]

        # point is within array
        return lerp(x[index], x[index + 1],
                    y[index], y[index + 1],
                    x0)

    else:

        # ensure x0 is always lower than x1
        if x1 < x0:
            temp = x0
            x0 = x1
            x1 = temp

        return integrate(x, y, x0, x1) / (x1 - x0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double maximum(double[:] data) nogil:
    """
    Return the maximum value in the buffer data.

    This is equivalent to Python's max() function, but is far faster
    for objects supporting the buffer interface where each element is a
    double.

    :param double data: Memoryview of a data array.
    :rtype: double
    """

    cdef:
        int i
        double result

    result = data[0]
    for i in range(1, data.shape[0]):
        result = max(result, data[i])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double minimum(double[:] data) nogil:
    """
    Return the minimum value in the buffer data.

    This is equivalent to Python's min() function, but is far faster
    for objects supporting the buffer interface where each element is a
    double.

    :param double data: Memoryview of a data array.
    :rtype: double
    """
    cdef:
        int i
        double result
    result = data[0]
    for i in range(1, data.shape[0]):
        result = min(result, data[i])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double peak_to_peak(double[:] data) nogil:
    """
    Return the peak-to-peak value in the buffer data.

    This is equivalent to Python's max() - min(), but is far faster
    for objects supporting the buffer interface where each element is a
    double.

    :param double data: Memoryview of a data array.
    :rtype: double
    """

    cdef:
        int i
        double result, max_, min_

    max_ = data[0]
    min_ = data[0]
    for i in range(1, data.shape[0]):
        max_ = max(max_, data[i])
        min_ = min(min_, data[i])
    result = max_ - min_
    return result


@cython.cdivision(True)
cdef bint solve_quadratic(double a, double b, double c, double *t0, double *t1) nogil:
    """
    Calculates the real roots of a quadratic equation.

    The a, b and c arguments are the three constants of the quadratic equation:

        f = a.x^2 + b.x^2 + c
        
    If the quadratic equation has 1 or 2 real roots, this function will return
    True. If there are no real roots this method will return False.
    
    The values of the real roots, are returned by setting the values of the
    memory locations pointed to by t0 and t1. In the case of a single root,
    both t0 and t1 will have the same value. If there are not roots, the values
    of t0 and t1 will be undefined. 

    :param double a: Quadratic constant. 
    :param double b: Quadratic constant.
    :param double c: Quadratic constant.
    :param double t0: 1st root of the quadratic.
    :param double t1: 2nd root of the quadratic.
    :return: True if real roots found, False otherwise.
    :rtype: bint
    """

    cdef double d, q

    # calculate discriminant
    d = b*b - 4*a*c

    # are there any real roots of the quadratic?
    if d < 0:
        return False

    # calculate roots using method described in the book:
    # "Physically Based Rendering - 2nd Edition", Elsevier 2010
    # this method is more numerically stable than the usual root equation
    if b < 0:
        q = -0.5 * (b - sqrt(d))
    else:
        q = -0.5 * (b + sqrt(d))
    t0[0] = q / a
    t1[0] = c / q
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint winding2d(double[:,::1] vertices) nogil:
    """
    Identifies the winding direction of a simple 2D polygon.

    Must be a simple polygon (none of the segments cross each other).
    This method is only valid for closed polygons,
    i.e. the first point is connected to the last point.

    Returns True if clockwise, false if anti-clockwise.

    Vertices must be a Nx2 array, this is not checked.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :rtype: bool
    """

    cdef:
        double sum = 0
        int i, length

    # Work out the signed area of the polygon (note: this is double the area because we need sign of magnitude
    # and can avoid dividing by 2).
    length = vertices.shape[0]
    for i in range(length - 1):
        sum += (vertices[i, 1] + vertices[i + 1, 1]) * (vertices[i + 1, 0] - vertices[i, 0])
    sum += (vertices[0, 1] + vertices[length - 1, 1]) * (vertices[0, 0] - vertices[length - 1, 0])
    return sum > 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint point_inside_polygon(double[:,::1] vertices, double ptx, double pty):
    """
    Cython utility for testing if a 2D point (ptx, pty) is inside a 2D polygon defined by
    the two memory views px_mv[:] and py_mv[:].

    This function implements the winding number method for testing if points are inside or
    outside an arbitrary polygon. Returns True if the test point is inside the specified
    polygon.

    .. WARNING:: For speed, this function does not perform any type or bounds
       checking. Supplying malformed data may result in data corruption or a
       segmentation fault.

    :param double vertices: Memory view of polygon's x,y coordinates with shape (N,2)
      where N is the number of points in the polygon.
    :param double ptx: the x coordinate of the test point.
    :param double pty: the y coordinate of the test point.
    :rtype: bool
    """

    cdef:
        int i, winding_number = 0
        double side

    for i in range(vertices.shape[0] - 1):

        # start case where first polygon edge y is less than test-point's y
        if vertices[i, 1] <= pty:
            # test for case of upward crossing
            if vertices[i+1, 1] > pty:

                # Test if point is on left side of line
                side = (vertices[i+1, 0] - vertices[i, 0]) * (pty - vertices[i, 1]) - (ptx -  vertices[i, 0]) * (vertices[i+1, 1] - vertices[i, 1])
                if side > 0:
                    winding_number += 1


        # else we must be considering case where first polygon edge point's y is greater than test point y
        else:
            if vertices[i+1, 1] <= pty:

                # Test if point is on right side of line
                side = (vertices[i+1, 0] - vertices[i, 0]) * (pty - vertices[i, 1]) - (ptx -  vertices[i, 0]) * (vertices[i+1, 1] - vertices[i, 1])
                if side < 0:
                    winding_number -= 1

    if winding_number == 0:
        return False
    else:
        return True


cdef int factorial(int n):
    """Calculate the factorial of an interger n (n!) through recursive calculation."""
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)


def _maximum(data):
    """Expose cython function for testing."""
    return maximum(data)


def _minimum(data):
    """Expose cython function for testing."""
    return minimum(data)


def _peak_to_peak(data):
    """Expose cython function for testing."""
    return peak_to_peak(data)


def _test_winding2d(p):
    """Expose cython function for testing."""
    return winding2d(p)


def _point_inside_polygon(vertices, ptx, pty):
    """Expose cython function for testing."""
    return point_inside_polygon(vertices, ptx, pty)
