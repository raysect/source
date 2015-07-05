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

from raysect.core.acceleration.boundingbox cimport BoundingBox
from raysect.core.classes cimport Ray
from raysect.core.math.point cimport Point

# c-structure that represent a kd-tree node
cdef struct kdnode:

    int type        # LEAF, X_AXIS, Y_AXIS, Z_AXIS
    double split    # split position
    int count       # upper index (BRANCH), item count (LEAF)
    int *items      # array of item ids


cdef class Item:

    cdef:
        readonly int id
        readonly BoundingBox box


cdef class _Edge:

    cdef:
        readonly int item
        readonly bint is_upper_edge
        readonly double value


cdef class KDTreeCore:

    cdef:
        kdnode *_nodes
        int _allocated_nodes
        int _next_node
        readonly BoundingBox bounds
        int _max_depth
        int _min_items
        double _hit_cost
        double _empty_bonus

    cdef int _build(self, list items, BoundingBox bounds, int depth=*)

    cdef tuple _split(self, list items, BoundingBox bounds)

    cdef list _get_edges(self, list items, int axis)

    cdef BoundingBox _get_lower_bounds(self, BoundingBox bounds, double split, int axis)

    cdef BoundingBox _get_upper_bounds(self, BoundingBox bounds, double split, int axis)

    cdef int _new_leaf(self, list items)

    cdef int _new_branch(self, tuple split_solution, int depth)

    cdef int _new_node(self)

    cpdef bint hit(self, Ray ray)

    cdef bint _hit(self, Ray ray)

    cdef bint _hit_node(self, int id, Ray ray, double min_range, double max_range)

    cdef bint _hit_branch(self, int id, Ray ray, double min_range, double max_range)

    cdef bint _hit_leaf(self, int id, Ray ray, double max_range)

    cpdef list contains(self, Point point)

    cdef list _contains(self, Point point)

    cdef list _contains_node(self, int id, Point point)

    cdef list _contains_branch(self, int id, Point point)

    cdef list _contains_leaf(self, int id, Point point)


cdef class KDTree(KDTreeCore):

    cdef bint _hit_leaf(self, int id, Ray ray, double max_range)

    cpdef bint _hit_items(self, list items, Ray ray, double max_range)

    cdef list _contains_leaf(self, int id, Point point)

    cpdef list _contains_items(self, list items, Point point)