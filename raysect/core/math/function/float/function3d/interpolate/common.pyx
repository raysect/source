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

import numpy as np
cimport numpy as np
from raysect.core.math.spatial.kdtree3d cimport Item3D
from raysect.core.boundingbox cimport BoundingBox3D, new_boundingbox3d
from raysect.core.math.point cimport Point3D, new_point3d
from raysect.core.math.cython cimport barycentric_inside_tetrahedra, barycentric_coords_tetra
cimport cython

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-6

# convenience defines
DEF V1 = 0
DEF V2 = 1
DEF V3 = 2
DEF V4 = 3

DEF X = 0
DEF Y = 1
DEF Z = 2


cdef class MeshKDTree3D(KDTree3DCore):

    def __init__(self, object vertices not None, object tetrahedra not None):

        vertices = np.array(vertices, dtype=np.double)
        tetrahedra = np.array(tetrahedra, dtype=np.int32)

        # check dimensions are correct
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("The vertex array must have dimensions Nx3.")

        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
            raise ValueError("The tetrahedra array must have dimensions Mx4.")

        # check tetrahedra contains only valid indices
        invalid = (tetrahedra[:, 0:4] < 0) | (tetrahedra[:, 0:4] >= vertices.shape[0])
        if invalid.any():
            raise ValueError("The tetrahedra array references non-existent vertices.")

        # assign to internal attributes
        self._vertices = vertices
        self._tetrahedra = tetrahedra

        # assign to memory views
        self._vertices_mv = vertices
        self._tetrahedra_mv = tetrahedra

        # initialise hit state attributes
        self.tetrahedra_id = -1
        self.i1 = -1
        self.i2 = -1
        self.i3 = -1
        self.i4 = -1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0

        # kd-Tree init
        items = []
        for tetrahedra in range(self._tetrahedra.shape[0]):
            items.append(Item3D(tetrahedra, self._generate_bounding_box(tetrahedra)))
        super().__init__(items, max_depth=0, min_items=1, hit_cost=50.0, empty_bonus=0.2)

        # todo: (possible enhancement) check if tetrahedra are overlapping?
        # (any non-owned vertex lying inside another tetrahedra)

        # init cache
        self._cache_available = False
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_z = 0.0
        self._cached_result = False

    def __getstate__(self):
        return self._tetrahedra, self._vertices, super().__getstate__()

    def __setstate__(self, state):

        self._tetrahedra, self._vertices, super_state = state
        super().__setstate__(super_state)

        # rebuild memory views
        self._vertices_mv = self._vertices
        self._tetrahedra_mv = self._tetrahedra

        # initialise hit state attributes
        self.tetrahedra_id = -1
        self.i1 = -1
        self.i2 = -1
        self.i3 = -1
        self.i4 = -1
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0

        # initialise cache values
        self._cache_available = False
        self._cached_x = 0.0
        self._cached_y = 0.0
        self._cached_z = 0.0
        self._cached_result = False

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef BoundingBox3D _generate_bounding_box(self, np.int32_t tetrahedra):
        """
        Generates a bounding box for the specified tetrahedra.

        A small degree of padding is added to the bounding box to provide the
        conservative bounds required by the watertight mesh algorithm.

        :param tetrahedra: tetrahedra array index.
        :return: A BoundingBox3D object.
        """

        cdef:
            np.int32_t i1, i2, i3, i4
            BoundingBox3D bbox

        i1 = self._tetrahedra_mv[tetrahedra, V1]
        i2 = self._tetrahedra_mv[tetrahedra, V2]
        i3 = self._tetrahedra_mv[tetrahedra, V3]
        i4 = self._tetrahedra_mv[tetrahedra, V4]

        bbox = new_boundingbox3d(
            new_point3d(
                min(self._vertices_mv[i1, X], self._vertices_mv[i2, X], self._vertices_mv[i3, X], self._vertices_mv[i4, X]),
                min(self._vertices_mv[i1, Y], self._vertices_mv[i2, Y], self._vertices_mv[i3, Y], self._vertices_mv[i4, Y]),
                min(self._vertices_mv[i1, Z], self._vertices_mv[i2, Z], self._vertices_mv[i3, Z], self._vertices_mv[i4, Z]),
            ),
            new_point3d(
                max(self._vertices_mv[i1, X], self._vertices_mv[i2, X], self._vertices_mv[i3, X], self._vertices_mv[i4, X]),
                max(self._vertices_mv[i1, Y], self._vertices_mv[i2, Y], self._vertices_mv[i3, Y], self._vertices_mv[i4, Y]),
                max(self._vertices_mv[i1, Z], self._vertices_mv[i2, Z], self._vertices_mv[i3, Z], self._vertices_mv[i4, Z]),
            ),
        )

        # The bounding box and tetrahedral vertices may not align following coordinate
        # transforms in the water tight mesh algorithm, therefore a small bit of padding
        # is added to avoid numerical representation issues.
        bbox.pad(max(BOX_PADDING, bbox.largest_extent() * BOX_PADDING))

        return bbox

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef bint _is_contained_leaf(self, np.int32_t id, Point3D point):

        cdef:
            np.int32_t index, tetrahedra, i1, i2, i3, i4
            double alpha, beta, gamma, delta

        # identify the first tetrahedra that contains the point, if any
        for index in range(self._nodes[id].count):

            # obtain vertex indices
            tetrahedra = self._nodes[id].items[index]
            i1 = self._tetrahedra_mv[tetrahedra, V1]
            i2 = self._tetrahedra_mv[tetrahedra, V2]
            i3 = self._tetrahedra_mv[tetrahedra, V3]
            i4 = self._tetrahedra_mv[tetrahedra, V4]

            barycentric_coords_tetra(self._vertices_mv[i1, X], self._vertices_mv[i1, Y], self._vertices_mv[i1, Z],
                                     self._vertices_mv[i2, X], self._vertices_mv[i2, Y], self._vertices_mv[i2, Z],
                                     self._vertices_mv[i3, X], self._vertices_mv[i3, Y], self._vertices_mv[i3, Z],
                                     self._vertices_mv[i4, X], self._vertices_mv[i4, Y], self._vertices_mv[i4, Z],
                                     point.x, point.y, point.z, &alpha, &beta, &gamma, &delta)

            if barycentric_inside_tetrahedra(alpha, beta, gamma, delta):

                # store id of tetrahedra hit
                self.tetrahedra_id = tetrahedra

                # store vertex indices and barycentric coords
                self.i1 = i1
                self.i2 = i2
                self.i3 = i3
                self.i4 = i4
                self.alpha = alpha
                self.beta = beta
                self.gamma = gamma
                self.delta = delta

                return True

        return False

    cpdef bint is_contained(self, Point3D point):
        """
        Traverses the kd-Tree to identify if the point is contained by an item.
        :param Point3D point: A Point3D object.
        :return: True if the point lies inside an item, false otherwise.
        """

        cdef bint result

        if self._cache_available and point.x == self._cached_x and point.y == self._cached_y and point.z == self._cached_z:
            return self._cached_result

        result = self._is_contained(point)

        # add cache
        self._cache_available = True
        self._cached_x = point.x
        self._cached_y = point.y
        self._cached_z = point.z
        self._cached_result = result

        return result
