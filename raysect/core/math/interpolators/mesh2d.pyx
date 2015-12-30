# cython: language_level=3

# Copyright (c) 2014-2015, Dr Alex Meakins, Raysect Project
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

from scipy.spatial import KDTree
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt

from raysect.core.math.function.function2d cimport Function2D
cimport cython

# convenience defines
DEF V1 = 0
DEF V2 = 1
DEF V3 = 2

DEF X = 0
DEF Y = 1

# todo: add docstrings

cdef class Interpolator2DMesh(Function2D):
    """
    An abstract data structure for interpolating data points lying on a triangular mesh.
    """

    cdef:
        double[:, ::1] _vertex_coords
        double[::1] _vertex_data
        np.int64_t[:, ::1] _triangles
        bint _limit
        double _default_value

    def __init__(self, object vertex_coords not None, object vertex_data not None, object triangles not None, bint limit=True, double default_value=0.0):
        """
        :param ndarray vertex_coords: An array of vertex coordinates with shape (num of vertices, 2). For each vertex
        there must be a (u, v) coordinate.
        :param ndarray vertex_data: An array of data points at each vertex with shape (num of vertices).
        :param ndarray triangles: An array of triangles with shape (num of triangles, 3). For each triangle, there must
        be three indices that identify the three corresponding vertices in vertex_coords that make up this triangle.
        """

        # convert to ndarrays for processing
        self._vertex_coords = np.array(vertex_coords, dtype=np.float64)
        self._vertex_data = np.array(vertex_data, dtype=np.float64)
        self._triangles = np.array(triangles, dtype=np.int64)

        # validate data
        # check sizes
        # check indices are in valid ranges

        # build kdtree
        # TODO: write me

        # check if triangles are overlapping
        # (any non-owned vertex lying inside another triangle)
        # TODO: write me (needs kdtree to be efficient)

        self._default_value = default_value
        self._limit = limit

    @classmethod
    def instance(cls, Interpolator2DMesh instance not None, object vertex_data=None, object limit=None, object default_value=None):

        cdef Interpolator2DMesh m

        m = Interpolator2DMesh.__new__(Interpolator2DMesh)

        # todo: update when kdtree added
        # copy source data
        m._vertex_coords = instance._vertex_coords
        m._triangles = instance._triangles
        # m._kdtree = instance._kdtree

        # do we have replacement vertex data?
        if vertex_data is None:
            m._vertex_data = instance._vertex_data
        else:
            m._vertex_data = np.array(vertex_data, dtype=np.float64)
            # TODO: validate

        # do we have a replacement limit check setting?
        if limit is None:
            m._limit = instance._limit
        else:
            m._limit = limit

        # do we have a replacement default value?
        if default_value is None:
            m._default_value = instance._default_value
        else:
            m._default_value = default_value

        return m

    cdef double evaluate(self, double x, double y) except *:

        cdef double alpha, beta, gamma

        # TODO: replace this with the kdtree, this is brute force and slow
        # check if point lies in any triangle and interpolate data inside the triangle it lies inside, if it does
        for triangle in range(self._triangles.shape[0]):
            self._calc_barycentric_coords(triangle, x, y, &alpha, &beta, &gamma)
            if self._contains(alpha, beta, gamma):
                return self._interpolate(triangle, x, y, alpha, beta, gamma)

        if not self._limit:
            return self._default_value

        raise ValueError("Requested value outside mesh bounds.")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline void _calc_barycentric_coords(self, np.int64_t triangle, double px, double py, double *alpha, double *beta, double *gamma):

        cdef:
            np.int64_t[:, ::1] triangles
            double[:, ::1] vertex_coords
            int i1, i2, i3
            double v1x, v2x, v3x, v1y, v2y, v3y
            double x1, x2, x3, y1, y2, y3
            double norm

        # cache locally to avoid pointless memory view checks
        triangles = self._triangles
        vertex_coords = self._vertex_coords

        # obtain vertex indices
        i1 = triangles[triangle, V1]
        i2 = triangles[triangle, V2]
        i3 = triangles[triangle, V3]

        # obtain the vertex coords
        v1x = vertex_coords[i1, X]
        v1y = vertex_coords[i1, Y]

        v2x = vertex_coords[i2, X]
        v2y = vertex_coords[i2, Y]

        v3x = vertex_coords[i3, X]
        v3y = vertex_coords[i3, Y]

        # compute common values
        x1 = v1x - v3x
        x2 = v3x - v2x
        x3 = px - v3x

        y1 = v1y - v3y
        y2 = v2y - v3y
        y3 = py - v3y

        norm = 1 / (x1 * y2 + y1 * x2)

        # compute barycentric coordinates
        alpha[0] = norm * (x2 * y3 + y2 * x3)
        beta[0] = norm * (x1 * y3 - y1 * x3)
        gamma[0] = 1.0 - alpha[0] - beta[0]

    cdef inline bint _contains(self, double alpha, double beta, double gamma):

        # Point is inside triangle if all coordinates lie in range [0, 1]
        # if all are > 0 then none can be > 1 from definition of barycentric coordinates
        return alpha >= 0 and beta >= 0 and gamma >= 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double _interpolate(self, int triangle, double px, double py, double alpha, double beta, double gamma):

        cdef:
            np.int64_t[:, ::1] triangles
            double[::1] vertex_data
            int i1, i2, i3
            double v1, v2, v3

        # cache locally to avoid pointless memory view checks
        triangles = self._triangles
        vertex_data = self._vertex_data

        # obtain the vertex indices
        i1 = triangles[triangle, V1]
        i2 = triangles[triangle, V2]
        i3 = triangles[triangle, V3]

        # obtain the vertex data
        v1 = vertex_data[i1]
        v2 = vertex_data[i2]
        v3 = vertex_data[i3]

        # barycentric interpolation
        return alpha * v1 + beta * v2 + gamma * v3
