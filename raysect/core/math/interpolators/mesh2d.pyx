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
from raysect.core.math.point cimport Point2D
cimport cython

# convenience defines
DEF V1 = 0
DEF V2 = 1
DEF V3 = 2

DEF X = 0
DEF Y = 1

# TODO: split instance into its own method (e.g. mymesh.instance(vertex_data=...)), current init interface is messy


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

    def __init__(self, object vertex_coords=None, object vertex_data=None, object triangles=None, object limit=None, object default_value=None, Interpolator2DMesh instance=None):
        """
        :param ndarray vertex_coords: An array of vertex coordinates with shape (num of vertices, 2). For each vertex
        there must be a (u, v) coordinate.
        :param ndarray vertex_data: An array of data points at each vertex with shape (num of vertices).
        :param ndarray triangles: An array of triangles with shape (num of triangles, 3). For each triangle, there must
        be three indices that identify the three corresponding vertices in vertex_coords that make up this triangle.
        """

        if instance is None:

            if vertex_coords is None or vertex_data is None or triangles is None:
                raise ValueError("At least vertex_coords, vertex_data and triangles or instance must be specified.")

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

            # None is used to identify when instances are permitted to pass through their default_value
            if default_value is None:
                self._default_value = 0.0
            else:
                self._default_value = default_value

            # None is used to identify when instances are permitted to pass through their limit setting
            if limit is None:
                self._limit = True
            else:
                self._limit = limit

        else:

            # todo: update when kdtree added
            # copy source data
            self._vertex_coords = instance._vertex_coords
            self._triangles = instance._triangles
            # self._kdtree = instance._kdtree

            # do we have replacement vertex data?
            if vertex_data is None:
                self._vertex_data = instance._vertex_data
            else:
                self._vertex_data = np.array(vertex_data, dtype=np.float64)

            # do we have a replacement default value?
            if default_value is None:
                self._default_value = instance._default_value
            else:
                self._default_value = default_value

            # do w have a replacement limit check setting?
            if limit is None:
                self._limit = instance._limit
            else:
                self._limit = limit

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





    #     self.vertices = np.zeros((vertex_coords.shape[0]), dtype=object)
    #     for index, vertex in enumerate(vertex_coords):
    #         self.vertices[index] = _Vertex2D(vertex[0], vertex[1], index)
    #
    #     self.vertex_data = np.array(vertex_data, dtype=np.float64)
    #
    #     self.triangles = np.zeros((triangles.shape[0]), dtype=object)
    #     for i, triangle in enumerate(triangles):
    #         try:
    #             v1 = _Vertex2D.all_vertices[triangle[0]]
    #             v2 = _Vertex2D.all_vertices[triangle[1]]
    #             v3 = _Vertex2D.all_vertices[triangle[2]]
    #         except IndexError:
    #             raise ValueError("vertex could not be found in vertex list")
    #
    #         # proper kd-tree for triangles will make this unnecessary, vertices don't need to store triangle refs.
    #         triangle = _Triangle2D(v1, v2, v3)
    #         v1.triangles.append(triangle)
    #         v2.triangles.append(triangle)
    #         v3.triangles.append(triangle)
    #
    #         self.triangles[i] = triangle
    #
    #     unique_vertices = [(vertex.u, vertex.v) for vertex in _Vertex2D.all_vertices]
    #
    #     # TODO - implement KD-tree here
    #     # construct KD-tree from vertices
    #     self.kdtree = KDTree(unique_vertices)
    #     self.kdtree_search = kdtree_search
    #
    # cdef double evaluate(self, double x, double y) except *:
    #     cdef _Triangle2D triangle
    #     triangle = self.find_triangle_containing(x, y)
    #     if triangle:
    #         return triangle.evaluate(x, y, self)
    #     return 0.0
    #
    # cpdef _Triangle2D find_triangle_containing(self, double u, double v):
    #     if self.kdtree_search:
    #         return self.kdtree_method(u, v)
    #     else:
    #         return self.brute_force_method(u, v)
    #
    # cdef _Triangle2D brute_force_method(self, double u, double v):
    #     cdef _Triangle2D triangle
    #     for triangle in self.triangles:
    #         if triangle.contains(u, v):
    #             return triangle
    #     return None
    #
    # cdef _Triangle2D kdtree_method(self, double u, double v):
    #     cdef:
    #         long[:] i_closest
    #         double[:] dist
    #         _Vertex2D closest_vertex
    #         _Triangle2D triangle
    #
    #     # Find closest vertex through KD-tree
    #     dist, i_closest = self.kdtree.query((u, v), k=10)
    #     closest_vertex = self.vertices[i_closest[0]]
    #
    #     # cycle through all triangles connected to this vertex
    #     for triangle in closest_vertex.triangles:
    #         if triangle.contains(u, v):
    #             return triangle
    #     return None
    #
    # cdef double get_vertex_data(self, int vertex_index):
    #     return self.vertex_data[vertex_index]
    #
    # def plot_mesh(self):
    #     plt.figure()
    #     for triangle in self.triangles:
    #         v1 = triangle.v1
    #         v2 = triangle.v2
    #         v3 = triangle.v3
    #         plt.plot([v1.u, v2.u, v3.u, v1.u], [v1.v, v2.v, v3.v, v1.v], color='b')
    #
    #     plt.axis('equal')
    #     plt.show()
    #
    # def copy_mesh_with_new_data(self, vertex_data):
    #     """
    #     Make a new TriangularMeshInterpolator2D from an existing instance, but with new vertex data.
    #
    #     :param ndarray vertex_data: An array of data points at each vertex with shape (num of vertices).
    #     """
    #
    #     # Make a new mesh object without invoking __init__()
    #     new_mesh = Interpolator2DMesh.__new__(Interpolator2DMesh)
    #
    #     # Copy over vertex data
    #     new_mesh.vertex_data = np.array(vertex_data, dtype=np.float64)
    #
    #     # Copy over other mesh attributes
    #     new_mesh.vertices = self.vertices
    #     new_mesh.triangles = self.triangles
    #     new_mesh.kdtree = self.kdtree
    #     new_mesh.kdtree_search = self.kdtree_search
    #     return new_mesh

