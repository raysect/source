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

from raysect.core cimport Ray, Intersection, Node, Primitive
from raysect.core.acceleration cimport BoundPrimitive


cdef class CSGPrimitive(Primitive):

    cdef CSGRoot _csgroot
    cdef BoundPrimitive _primitive_a
    cdef BoundPrimitive _primitive_b
    cdef Ray _cache_ray
    cdef Intersection _cache_intersection_a
    cdef Intersection _cache_intersection_b
    cdef Intersection _cache_last_intersection
    cdef bint _cache_invalid

    cdef bint terminate_early(self, Intersection intersection)

    cdef Intersection _identify_intersection(self, Ray ray, Intersection intersection_a, Intersection intersection_b, Intersection closest_intersection)

    cdef Intersection _closest_intersection(self, Intersection a, Intersection b)

    cdef bint _valid_intersection(self, Intersection a, Intersection b, Intersection closest)

    cdef Intersection _modify_intersection(self, Intersection closest, Intersection a, Intersection b)

    cdef void rebuild(self)


cdef class CSGRoot(Node):

    cdef CSGPrimitive csg_primitive


cdef class Union(CSGPrimitive):

    pass


cdef class Intersect(CSGPrimitive):

    pass


cdef class Subtract(CSGPrimitive):

    pass



