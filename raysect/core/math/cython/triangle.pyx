# cython: language_level=3

# Copyright (c) 2014-17, Dr Alex Meakins, Raysect Project
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


cdef inline bint inside_triangle(double v1x, double v1y, double v2x, double v2y,
                                 double v3x, double v3y, double px, double py) nogil:
    """Returns True if test point is inside triangle."""

    cdef:
        double ux, uy, vx, vy

    # calculate vectors
    ux = v2x - v1x
    uy = v2y - v1y

    vx = px - v1x
    vy = py - v1y

    # calculate z component of cross product of vectors between vertices
    # vertex is convex if z component of u.cross(v) is negative
    if (ux * vy - vx * uy) > 0:
        return False

    # calculate vectors
    ux = v3x - v2x
    uy = v3y - v2y

    vx = px - v2x
    vy = py - v2y

    # calculate z component of cross product of vectors between vertices
    # vertex is convex if z component of u.cross(v) is negative
    if (ux * vy - vx * uy) > 0:
        return False

    # calculate vectors
    ux = v1x - v3x
    uy = v1y - v3y

    vx = px - v3x
    vy = py - v3y

    # calculate z component of cross product of vectors between vertices
    # vertex is convex if z component of u.cross(v) is negative
    if (ux * vy - vx * uy) > 0:
        return False

    return True


def _test_inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, px, py):
    """Expose cython function for testing."""
    return inside_triangle(v1x, v1y, v2x, v2y, v3x, v3y, px, py)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void calc_barycentric_coords(Point2D v1, Point2D v2, Point2D v3, Point2D test_point,
                                         double *alpha, double *beta, double *gamma) nogil:
    """
    Cython utility for calculating the barycentric coordinates of a test point.

    :param Point2D v1: Triangle vertex 1.
    :param Point2D v2: Triangle vertex 2.
    :param Point2D v3: Triangle vertex 3.
    :param Point2D test_point: The point for which barycentric coordinates
      should be calculated.
    :param double alpha: returned coordinate alpha
    :param double beta: returned coordinate beta
    :param double gamma: returned coordinate gamma
    """

    cdef:
        double x1, x2, x3, y1, y2, y3
        double norm

    # compute common values
    x1 = v1.x - v3.x
    x2 = v3.x - v2.x
    x3 = test_point.x - v3.x

    y1 = v1.y - v3.y
    y2 = v2.y - v3.y
    y3 = test_point.y - v3.y

    norm = 1 / (x1 * y2 + y1 * x2)

    # compute barycentric coordinates
    alpha[0] = norm * (x2 * y3 + y2 * x3)
    beta[0] = norm * (x1 * y3 - y1 * x3)
    gamma[0] = 1.0 - alpha[0] - beta[0]
