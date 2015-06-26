# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.scenegraph.primitive cimport Primitive
from raysect.core.acceleration.boundingbox cimport BoundingBox
from raysect.core.math.point cimport Point
from raysect.core.math.normal cimport Normal
from raysect.core.math.vector cimport Vector
from raysect.core.classes cimport Ray

cdef class _Edge:

    cdef:
        readonly double value
        readonly bint is_upper_edge
        readonly Triangle triangle


cdef class _Node:

    cdef readonly _Node lower_branch
    cdef readonly _Node upper_branch
    cdef readonly list triangles
    cdef readonly int axis
    cdef readonly double split
    cdef readonly bint is_leaf

    cdef object build(self, BoundingBox node_bounds, list triangles, int depth, int min_triangles, double hit_cost)

    cdef void _become_leaf(self, list triangles)

    cdef list _build_edges(self, list triangles, int axis)

    cdef BoundingBox _calc_lower_bounds(self, BoundingBox node_bounds, double split_value, int axis)

    cdef BoundingBox _calc_upper_bounds(self, BoundingBox node_bounds, double split_value, int axis)

    cdef tuple hit(self, Ray ray, double min_range, double max_range)

    cdef inline tuple _hit_branch(self, Ray ray, double min_range, double max_range)

    cdef inline tuple _hit_leaf(self, Ray ray, double max_range)

    cdef tuple _calc_rayspace_transform(self, Ray ray)

    cdef tuple _hit_triangle(self, Triangle triangle, tuple ray_transform, Ray ray)

    cdef list contains(self, Point point)


cdef class Triangle:

    cdef:
        readonly Point v1, v2, v3
        readonly Normal n1, n2, n3
        readonly Normal face_normal
        readonly bint _smoothing_enabled

    cdef Normal _calc_face_normal(self)

    cpdef Normal interpolate_normal(self, double u, double v, double w, bint smoothing=*)


cdef class Mesh(Primitive):

    cdef:
        readonly list triangles
        public bint smoothing
        public int kdtree_max_depth
        public int kdtree_min_triangles
        public double kdtree_hit_cost
        BoundingBox _world_box
        _Node _kdtree

        cdef object _build_kdtree(self)

        cdef object _build_world_box(self)


