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

from raysect.core.boundingbox cimport BoundingBox3D
from raysect.core.ray cimport Ray
from raysect.core.math.point cimport Point3D
from libc.stdint cimport int32_t

# c-structure that represent a kd-tree node
cdef struct kdnode:

    int32_t type        # LEAF, X_AXIS, Y_AXIS, Z_AXIS
    double split        # split position
    int32_t count       # upper index (BRANCH), item count (LEAF)
    int32_t *items      # array of item ids


cdef struct edge:

    bint is_upper_edge
    double value


cdef class Item3D:

    cdef:
        readonly int32_t id
        readonly BoundingBox3D box


cdef class KDTree3DCore:

    cdef:
        kdnode *_nodes
        int32_t _allocated_nodes
        int32_t _next_node
        readonly BoundingBox3D bounds
        int32_t _max_depth
        int32_t _min_items
        double _hit_cost
        double _empty_bonus

    cdef int32_t _build(self, list items, BoundingBox3D bounds, int32_t depth=*)

    cdef tuple _split(self, list items, BoundingBox3D bounds)

    cdef void _get_edges(self, list items, int32_t axis, int32_t *num_edges, edge **edges_ptr)

    cdef void _free_edges(self, edge **edges_ptr)

    cdef BoundingBox3D _get_lower_bounds(self, BoundingBox3D bounds, double split, int32_t axis)

    cdef BoundingBox3D _get_upper_bounds(self, BoundingBox3D bounds, double split, int32_t axis)

    cdef int32_t _new_leaf(self, list ids)

    cdef int32_t _new_branch(self, tuple split_solution, int32_t depth)

    cdef int32_t _new_node(self)
    
    cpdef bint is_contained(self, Point3D point)

    cdef bint _is_contained(self, Point3D point)

    cdef bint _is_contained_node(self, int32_t id, Point3D point)

    cdef bint _is_contained_branch(self, int32_t id, Point3D point)

    cdef bint _is_contained_leaf(self, int32_t id, Point3D point)

    cpdef bint trace(self, Ray ray)

    cdef bint _trace(self, Ray ray)

    cdef bint _trace_node(self, int32_t id, Ray ray, double min_range, double max_range)

    cdef bint _trace_branch(self, int32_t id, Ray ray, double min_range, double max_range)

    cdef bint _trace_leaf(self, int32_t id, Ray ray, double max_range)

    cpdef list items_containing(self, Point3D point)

    cdef list _items_containing(self, Point3D point)

    cdef list _items_containing_node(self, int32_t id, Point3D point)

    cdef list _items_containing_branch(self, int32_t id, Point3D point)

    cdef list _items_containing_leaf(self, int32_t id, Point3D point)

    cdef void _reset(self)

    cdef double _read_double(self, object file)

    cdef int32_t _read_int32(self, object file)


cdef class KDTree3D(KDTree3DCore):

    cdef bint _trace_leaf(self, int32_t id, Ray ray, double max_range)

    cpdef bint _trace_items(self, list items, Ray ray, double max_range)

    cdef list _items_containing_leaf(self, int32_t id, Point3D point)

    cpdef list _items_containing_items(self, list items, Point3D point)