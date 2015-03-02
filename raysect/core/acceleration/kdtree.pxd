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

from raysect.core.acceleration.accelerator cimport Accelerator
from raysect.core.acceleration.boundprimitive cimport BoundPrimitive
from raysect.core.acceleration.boundingbox cimport BoundingBox
from raysect.core.classes cimport Ray
from raysect.core.math.point cimport Point
from raysect.core.classes cimport Intersection

cdef class _Edge:

    cdef readonly double value
    cdef readonly bint is_upper_edge
    cdef readonly BoundPrimitive primitive


cdef class _Node:

    cdef readonly _Node lower_branch
    cdef readonly _Node upper_branch
    cdef readonly list primitives
    cdef readonly int axis
    cdef readonly double split
    cdef readonly bint is_leaf

    cdef object build(self, BoundingBox node_bounds, list primitives, int depth, int min_primitives, double hit_cost)

    cdef void _become_leaf(self, list primitives)

    cdef list _build_edges(self, list primitives, int axis)

    cdef BoundingBox _calc_lower_bounds(self, BoundingBox node_bounds, double split_value, int axis)

    cdef BoundingBox _calc_upper_bounds(self, BoundingBox node_bounds, double split_value, int axis)

    cdef Intersection hit(self, Ray ray, double min_range, double max_range)

    cdef inline Intersection _hit_leaf(self, Ray ray, double max_range)

    cdef inline Intersection _hit_branch(self, Ray ray, double min_range, double max_range)

    cdef list contains(self, Point point)


cdef class KDTree(Accelerator):

    cdef readonly BoundingBox world_box
    cdef readonly _Node tree
    cdef public int max_depth
    cdef public double hit_cost
    cdef public int min_primitives

    cpdef Intersection hit(self, Ray ray)

    cpdef list contains(self, Point point)

    cdef list _bound_primitives(self, list primitives)

    cdef void _build_world_box(self, list primitives)



