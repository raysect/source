# cython: language_level=3

# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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

import numpy as np
cimport numpy as np
from raysect.core.math.spatial.kdtree2d cimport Item2D
from raysect.core.boundingbox cimport BoundingBox2D, new_boundingbox2d
from raysect.core.math.point cimport Point2D, new_point2d
from raysect.core.math.cython cimport barycentric_inside_triangle, barycentric_coords
cimport cython

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-6

# convenience defines
DEF V1 = 0
DEF V2 = 1
DEF V3 = 2

DEF X = 0
DEF Y = 1


cdef class MeshKDTree2D(KDTree2DCore):

    def __init__(self, object vertices not None, object triangles not None):

        self._vertices = vertices
        self._triangles = triangles

        # check dimensions are correct
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("The vertex array must have dimensions Nx2.")

        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("The triangle array must have dimensions Mx3.")

        # check triangles contains only valid indices
        invalid = (triangles[:, 0:3] < 0) | (triangles[:, 0:3] >= vertices.shape[0])
        if invalid.any():
            raise ValueError("The triangle array references non-existent vertices.")

        # kd-Tree init
        items = []
        for triangle in range(self._triangles.shape[0]):
            items.append(Item2D(triangle, self._generate_bounding_box(triangle)))
        super().__init__(items, max_depth=0, min_items=1, hit_cost=50.0, empty_bonus=0.2)

        # todo: (possible enhancement) check if triangles are overlapping?
        # (any non-owned vertex lying inside another triangle)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef BoundingBox2D _generate_bounding_box(self, np.int32_t triangle):
        """
        Generates a bounding box for the specified triangle.

        A small degree of padding is added to the bounding box to provide the
        conservative bounds required by the watertight mesh algorithm.

        :param triangle: Triangle array index.
        :return: A BoundingBox2D object.
        """

        cdef:
            np.int32_t i1, i2, i3
            BoundingBox2D bbox

        i1 = self._triangles[triangle, V1]
        i2 = self._triangles[triangle, V2]
        i3 = self._triangles[triangle, V3]

        bbox = new_boundingbox2d(
            new_point2d(
                min(self._vertices[i1, X], self._vertices[i2, X], self._vertices[i3, X]),
                min(self._vertices[i1, Y], self._vertices[i2, Y], self._vertices[i3, Y]),
            ),
            new_point2d(
                max(self._vertices[i1, X], self._vertices[i2, X], self._vertices[i3, X]),
                max(self._vertices[i1, Y], self._vertices[i2, Y], self._vertices[i3, Y]),
            ),
        )

        # The bounding box and triangle vertices may not align following coordinate
        # transforms in the water tight mesh algorithm, therefore a small bit of padding
        # is added to avoid numerical representation issues.
        bbox.pad(max(BOX_PADDING, bbox.largest_extent() * BOX_PADDING))

        return bbox

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef bint _is_contained_leaf(self, np.int32_t id, Point2D point):

        cdef:
            np.int32_t index, triangle, i1, i2, i3
            double alpha, beta, gamma

        # identify the first triangle that contains the point, if any
        for index in range(self._nodes[id].count):

            # obtain vertex indices
            triangle = self._nodes[id].items[index]
            i1 = self._triangles[triangle, V1]
            i2 = self._triangles[triangle, V2]
            i3 = self._triangles[triangle, V3]

            barycentric_coords(self._vertices[i1, X], self._vertices[i1, Y],
                               self._vertices[i2, X], self._vertices[i2, Y],
                               self._vertices[i3, X], self._vertices[i3, Y],
                               point.x, point.y, &alpha, &beta, &gamma)

            if barycentric_inside_triangle(alpha, beta, gamma):

                # store id of triangle hit
                self.triangle_id = triangle

                # store vertex indices and barycentric coords
                self.i1 = i1
                self.i2 = i2
                self.i3 = i3
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma

                return True

        return False