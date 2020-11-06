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


cdef bint inside_tetrahedra(double v1x, double v1y, double v1z,
                            double v2x, double v2y, double v2z,
                            double v3x, double v3y, double v3z,
                            double v4x, double v4y, double v4z,
                            double px, double py, double pz) nogil:
    """
    Cython utility for testing if point is inside a tetrahedra.

    Note if you have barycentric coordinates available, it is quicker to use
    barycentric_inside_tetrahedra().

    :param double v1x: x coord of tetrahedra vertex 1.
    :param double v1y: y coord of tetrahedra vertex 1.
    :param double v1z: z coord of tetrahedra vertex 1.
    :param double v2x: x coord of tetrahedra vertex 2.
    :param double v2y: y coord of tetrahedra vertex 2.
    :param double v2z: z coord of tetrahedra vertex 2.
    :param double v3x: x coord of tetrahedra vertex 3.
    :param double v3y: y coord of tetrahedra vertex 3.
    :param double v3z: z coord of tetrahedra vertex 3.
    :param double v4x: x coord of tetrahedra vertex 4.
    :param double v4y: y coord of tetrahedra vertex 4.
    :param double v4z: z coord of tetrahedra vertex 4.    
    
    :param double px: x coord of test point.
    :param double py: y coord of test point.
    :param double pz: z coord of test point.
    :return: True if point is inside tetrahedra, False otherwise.
    :rtype: bool
    """

    cdef:
        double ux, uy, uz, vx, vy, vz, wx, wy, wz, tx, ty, tz

    # calculate vectors
    ux = v4x - v1x
    uy = v4y - v1y
    uz = v4z - v1z

    vx = v2x - v1x
    vy = v2y - v1y
    vz = v2z - v1z

    tx = px - v1x
    ty = py - v1y
    tz = pz - v1z

    # calculate cross product of vectors between u & v
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx


    # calculate inner product of vectors betwee, w & t
    # point is outside of surface if w.dot(t) is negative
    if (wx * tx + wy * ty + wz * tz) < 0:
        return False

    # calculate vectors
    ux = v4x - v2x
    uy = v4y - v2y
    uz = v4z - v2z

    vx = v3x - v2x
    vy = v3y - v2y
    vz = v3z - v2z

    tx = px - v2x
    ty = py - v2y
    tz = pz - v2z

    # calculate cross product of vectors between u & v
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx


    # calculate inner product of vectors betwee, w & t
    # point is outside of surface if w.dot(t) is negative
    if (wx * tx + wy * ty + wz * tz) < 0:
        return False

    # calculate vectors
    ux = v4x - v3x
    uy = v4y - v3y
    uz = v4z - v3z

    vx = v1x - v3x
    vy = v1y - v3y
    vz = v1z - v3z

    tx = px - v3x
    ty = py - v3y
    tz = pz - v3z

    # calculate cross product of vectors between u & v
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx


    # calculate inner product of vectors betwee, w & t
    # point is outside of surface if w.dot(t) is negative
    if (wx * tx + wy * ty + wz * tz) < 0:
        return False

    # calculate vectors
    ux = v2x - v1x
    uy = v2y - v1y
    uz = v2z - v1z

    vx = v3x - v1x
    vy = v3y - v1y
    vz = v3z - v1z

    tx = px - v1x
    ty = py - v1y
    tz = pz - v1z

    # calculate cross product of vectors between u & v
    wx = uy * vz - uz * vy
    wy = uz * vx - ux * vz
    wz = ux * vy - uy * vx


    # calculate inner product of vectors betwee, w & t
    # point is outside of surface if w.dot(t) is negative
    if (wx * tx + wy * ty + wz * tz) < 0:
        return False

    return True


def _test_inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z,
                                   v4x, v4y, v4z, px, py, pz):
    """Expose cython function for testing."""
    return inside_tetrahedra(v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z,
                                   v4x, v4y, v4z, px, py, pz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void barycentric_coords_tetra(double v1x, double v1y, double v1z,
                                   double v2x, double v2y, double v2z,
                                   double v3x, double v3y, double v3z,
                                   double v4x, double v4y, double v4z,
                                   double px, double py, double pz,
                                   double *alpha, double *beta, double *gamma, double *delta) nogil:
    """
    Cython utility for calculating the barycentric coordinates of a test point.

    :param double v1x: x coord of tetrahedra vertex 1.
    :param double v1y: y coord of tetrahedra vertex 1.
    :param double v1z: z coord of tetrahedra vertex 1.
    :param double v2x: x coord of tetrahedra vertex 2.
    :param double v2y: y coord of tetrahedra vertex 2.
    :param double v2z: z coord of tetrahedra vertex 2.
    :param double v3x: x coord of tetrahedra vertex 3.
    :param double v3y: y coord of tetrahedra vertex 3.
    :param double v3z: z coord of tetrahedra vertex 3.
    :param double v4x: x coord of tetrahedra vertex 4.
    :param double v4y: y coord of tetrahedra vertex 4.
    :param double v4z: z coord of tetrahedra vertex 4.    
    
    :param double px: x coord of test point.
    :param double py: y coord of test point.
    :param double pz: z coord of test point.
    :param double* alpha: returned coordinate alpha.
    :param double* beta: returned coordinate beta.
    :param double* gamma: returned coordinate gamma.
    :param double* delta: returned coordinate delta.
    """

    cdef:
        double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4
        double norm

    # compute common values
    x1 = v1x - v4x
    x2 = v2x - v4x
    x3 = v3x - v4x
    x4 = px - v4x

    y1 = v1y - v4y
    y2 = v2y - v4y
    y3 = v3y - v4y
    y4 = py - v4y

    z1 = v1z - v4z
    z2 = v2z - v4z
    z3 = v3z - v4z
    z4 = pz - v4z

    norm = 1 / (x1 * y2 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1 - x2 * y1 * z3 - x1 * y3 * z2)

    # compute barycentric coordinates
    alpha[0] = norm * ((y2 * z3 - y3 * z2) * x4 - (x2 * z3 - x3 * z2) * y4 + (x2 * y3 - x3 * y2) * z4)
    beta[0] = norm * (-1 * (y1 * z3 - y3 * z1) * x4 + (x1 * z3 - x3 * z1) * y4 - (x1 * y3 - x3 * y1) * z4)
    gamma[0] = norm * ((y1 * z2 - y2 * z1) * x4 - (x1 * z2 - x2 * z1) * y4 + (x1 * y2 - x2 * y1) * z4)
    delta[0] = 1.0 - alpha[0] - beta[0] - gamma[0]


cdef bint barycentric_inside_tetrahedra(double alpha, double beta, double gamma, double delta) nogil:
    """
    Cython utility for testing if a barycentric point lies inside a tetrahedra.

    :param double alpha: barycentric coordinate alpha.
    :param double beta: barycentric coordinate beta.
    :param double gamma: barycentric coordinate gamma.
    :param double delta: barycentric coordinate delta.
    :rtype: bool
    """

    # Point is inside tetrahedra if all coordinates lie in range [0, 1]
    # if all are > 0 then none can be > 1 from definition of barycentric coordinates
    return alpha >= 0 and beta >= 0 and gamma >= 0 and delta >= 0


cdef double barycentric_interpolation_tetra(double alpha, double beta, double gamma, double delta,
                                            double va, double vb, double vc, double vd) nogil:
    """
    Cython utility for interpolation of data at tetrahedra vertices.

    :param double alpha: Vertex 1 barycentric coordinate.
    :param double beta: Vertex 2 barycentric coordinate.
    :param double gamma: Vertex 3 barycentric coordinate.
    :param double delta: Vertex 4 barycentric coordinate.
    :param double va: Data point at Vertex 1.
    :param double vb: Data point at Vertex 2.
    :param double vc: Data point at Vertex 3.
    :param double vd: Data point at Vertex 4.
    :rtype: double
    """
    return alpha * va + beta * vb + gamma * vc + delta * vd
